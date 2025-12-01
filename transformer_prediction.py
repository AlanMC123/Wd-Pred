#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构版 Transformer 多输入预测脚本（含 Wordle grid 特征，WandB-safe 日志）
使用 Transformer Encoder 代替 LSTM 处理历史序列。

直接运行即开始训练（默认 RUN_MODE="train"）。

*** 1. 修复了网格特征的数据泄露问题。 ***
*** 2. 将玩家猜词过程编码为时间序列，并使用 Transformer 独立处理。 ***
"""

import os
import json
import random
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Tuple, List
from tensorflow.keras.layers import (Input, Dense, Dropout, Embedding, Flatten,
                                     LayerNormalization, GlobalAveragePooling1D,
                                     MultiHeadAttention, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, roc_curve

# ==========================================================
# 全局配置
# ==========================================================

TRAIN_FILE = "dataset/train_data.csv"
VAL_FILE = "dataset/val_data.csv"
TEST_FILE = "dataset/test_data.csv"

DIFFICULTY_FILE = "dataset/difficulty.csv"
PLAYER_FILE = "dataset/player_data.csv"

LOOK_BACK = 5
BATCH_SIZE = 1024
EPOCHS = 15
# Transformer 架构参数
LEARNING_RATE = 0.0005 

MODEL_SAVE_PATH = "models/transformer/transformer_model.keras"
TOKENIZER_PATH = "models/transformer/transformer_tokenizer.json"

# Transformer 核心参数
PROJECT_DIM = 32         # 序列特征投影维度 (用于Transformer内部)
NUM_HEADS = 4            # 多头注意力头数
FF_DIM = 32              # 前馈网络维度
TRANSFORMER_LAYERS = 2   # Transformer Encoder 层数
DROPOUT_RATE = 0.15
EMBEDDING_DIM = 24       # 单词ID嵌入维度

OOV_TOKEN = "<OOV>"

LARGE_ERROR_THRESHOLD = 1.5
PATIENCE = 5
REPORT_SAVE_PATH = "outputs/transformer_output.txt"

# 固定随机种子
SEED = 202

# Wordle固定参数
MAX_TRIES = 6  # 最大尝试次数
GRID_FEAT_LEN = 8 # 每次猜词的特征维度：3 (G/Y/R 累计) + 5 (Pos-G 累计)

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def ensure_dirs():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models/transformer", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing: {path}")
    return pd.read_csv(path, usecols=usecols)

# --------------------------
# Wordle 猜词过程序列编码 (新函数)
# --------------------------
def encode_guess_sequence(grid_cell):
    """
    解析 Wordle 网格文本，将猜词过程编码为时间序列。
    返回 shape (MAX_TRIES, GRID_FEAT_LEN) 的特征序列。
    每个时间步 t 的特征是截止到 guess t 的累积特征。
    """
    # 默认返回 (6, 8) 零矩阵
    default_seq = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)

    if pd.isna(grid_cell):
        return default_seq

    try:
        if isinstance(grid_cell, (list, tuple)):
            grid_list = list(grid_cell)
        else:
            # 安全地将字符串解析为列表
            grid_list = ast.literal_eval(grid_cell)
            if not isinstance(grid_list, (list, tuple)):
                grid_list = [grid_list]
            grid_list = [str(r) for r in grid_list if isinstance(r, (str, bytes))]
    except Exception:
        return default_seq
    
    # 实际进行的尝试次数
    num_rows = len(grid_list)
    
    # 累积统计变量 (重置)
    cumulative_greens = 0
    cumulative_yellows = 0
    cumulative_grays = 0
    cumulative_pos_green_counts = np.zeros(5, dtype=np.float32)

    # 结果序列
    feature_sequence = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)

    # 归一化基数
    norm_base_cells = float(MAX_TRIES * 5) # 总单元格数 (30)
    norm_base_rows = float(MAX_TRIES)      # 总行数 (6)
    
    for t in range(MAX_TRIES):
        # 只有在 t < num_rows 时，才进行累积
        if t < num_rows:
            # 统计当前行的特征
            row = grid_list[t]
            greens_t, yellows_t, grays_t = 0, 0, 0
            pos_green_counts_t = np.zeros(5, dtype=np.float32)
            
            if isinstance(row, str) and len(row) == 5:
                for i, ch in enumerate(row):
                    if ch == "🟩":
                        greens_t += 1
                        pos_green_counts_t[i] += 1.0
                    elif ch == "🟨":
                        yellows_t += 1
                    elif ch == "⬜" or ch == "⬛":
                        grays_t += 1
            
            # 累积到总数
            cumulative_greens += greens_t
            cumulative_yellows += yellows_t
            cumulative_grays += grays_t
            cumulative_pos_green_counts += pos_green_counts_t
            
        # 构建当前时间步 t 的特征向量 (使用累积量)
        feat = np.zeros(GRID_FEAT_LEN, dtype=np.float32)
        
        # 归一化
        feat[0] = cumulative_greens / norm_base_cells
        feat[1] = cumulative_yellows / norm_base_cells
        feat[2] = cumulative_grays / norm_base_cells
        
        # 位置绿块归一化
        for i in range(5):
            feat[3 + i] = cumulative_pos_green_counts[i] / norm_base_rows 
            
        feature_sequence[t] = feat
        
    return feature_sequence


# --------------------------
# Tokenizer
# --------------------------
def fit_tokenizer(train_df):
    tokenizer = Tokenizer(oov_token=OOV_TOKEN, filters='', lower=True)
    tokenizer.fit_on_texts(train_df["target"].astype(str))
    # 更新保存路径
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenizer.word_index, f, indent=2)
    return tokenizer

def load_tokenizer():
    # 更新加载路径
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    tk = Tokenizer(oov_token=OOV_TOKEN)
    tk.word_index = word_index
    return tk

# --------------------------
# 特征附加（包含 guess sequence）
# --------------------------
def attach_features(df, tokenizer, diff_map, user_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)

    if "processed_text" in df.columns:
        # 使用新的序列编码函数
        df["guess_seq_feat"] = df["processed_text"].apply(encode_guess_sequence)
    else:
        # 更新默认形状
        df["guess_seq_feat"] = [np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32) for _ in range(len(df))]

    return df

# --------------------------
# 历史建表 and 滑窗生成样本
# --------------------------
def build_history(df) -> Dict[str, List[Tuple]]:
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        # 存储元组: (Trial, word_difficulty, word_id, user_bias, guess_seq_feat)
        hist[u] = [(int(r["Trial"]),
                    float(r["word_difficulty"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    np.array(r["guess_seq_feat"], dtype=np.float32)) # <-- guess_seq_feat (6, 8) 数组
                   for _, r in g.iterrows()]
    return hist

def create_samples(history, look_back):
    # X_grid 现在代表 guess sequence X_guess_seq
    X_seq, X_diff, X_wid, X_bias, X_guess_seq, y_steps, y_succ = [], [], [], [], [], [], []
    for user, events in history.items():
        if len(events) <= look_back:
            continue
        for i in range(look_back, len(events)):
            window = events[i-look_back:i] # 历史窗口 (i-LOOK_BACK 到 i-1)
            target = events[i]             # 目标事件 (第 i 局游戏)

            # 历史序列特征 (基于 window)
            trials = np.array([t[0] for t in window], np.float32)
            norm = trials / 7.0
            std = np.std(trials) / 7.0
            seq = np.stack([norm, np.full_like(norm, std)], axis=1) # Shape: (LOOK_BACK, 2)
            X_seq.append(seq)
            
            # 目标事件的先验特征 (基于 target)
            X_diff.append([target[1] / 7.0])
            X_wid.append([target[2]])
            X_bias.append([target[3] / 7.0])
            
            # 网格序列特征 (基于 window)
            # FIX: 使用历史窗口中最后一个事件 (i-1) 的 guess sequence feature 来预测第 i 个事件
            X_guess_seq.append(window[-1][4]) 

            # 目标输出 (基于 target)
            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)

    if not X_seq:
        return (np.zeros((0, look_back, 2), np.float32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, 1), np.int32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, MAX_TRIES, GRID_FEAT_LEN), np.float32), # <-- 更新默认形状 (0, 6, 8)
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32))

    return (
        np.array(X_seq, np.float32),
        np.array(X_diff, np.float32),
        np.array(X_wid, np.int32),
        np.array(X_bias, np.float32),
        np.array(X_guess_seq, np.float32), # Shape is now (N, 6, 8)
        np.array(y_steps, np.float32),
        np.array(y_succ, np.float32)
    )

# ==========================================================
# Transformer Block Layer
# ==========================================================
class TransformerBlock(tf.keras.layers.Layer):
    """标准的 Transformer Encoder Block"""
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        # Attention
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        # Feed Forward
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)


# ==========================================================
# Transformer 模型
# ==========================================================
def build_model(look_back, vocab_size, project_dim=PROJECT_DIM, num_heads=NUM_HEADS, 
                ff_dim=FF_DIM, n_layers=TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE):
    
    # 历史序列输入分支 (Sequential Input: LOOK_BACK, 2)
    h_in = Input((look_back, 2), name="input_history")
    
    # 1. Feature Projection (2 -> PROJECT_DIM)
    x = Dense(project_dim, activation="relu")(h_in)
    
    # 2. Positional Encoding (Simplified additive encoding)
    positions = tf.range(start=0, limit=look_back, delta=1)
    pos_emb = Embedding(input_dim=look_back, output_dim=project_dim)(positions)
    x = x + pos_emb
    
    # 3. Transformer Blocks
    for _ in range(n_layers):
        x = TransformerBlock(project_dim, num_heads, ff_dim, dropout_rate)(x)
    
    # 4. Pooling
    x = GlobalAveragePooling1D()(x) # 汇聚序列信息
    x = Dropout(dropout_rate)(x)
    
    # --- 猜词序列输入分支 (新分支) ---
    # Wordle guess sequence 输入分支 (Sequential Input: MAX_TRIES, GRID_FEAT_LEN)
    guess_seq_in = Input((MAX_TRIES, GRID_FEAT_LEN), name="input_guess_sequence")

    # 1. Feature Projection (GRID_FEAT_LEN -> PROJECT_DIM)
    g = Dense(project_dim, activation="relu")(guess_seq_in)
    
    # 2. Positional Encoding
    positions_g = tf.range(start=0, limit=MAX_TRIES, delta=1)
    # MAX_TRIES = 6，所以这里只需要 6 个位置编码
    pos_emb_g = Embedding(input_dim=MAX_TRIES, output_dim=project_dim)(positions_g)
    g = g + pos_emb_g
    
    # 3. Transformer Blocks for Guess Sequence
    for _ in range(n_layers):
        g = TransformerBlock(project_dim, num_heads, ff_dim, dropout_rate)(g)
    
    # 4. Pooling
    g = GlobalAveragePooling1D()(g) # 汇聚猜词序列信息
    g = Dropout(dropout_rate)(g)
    # --- 猜词序列输入分支结束 ---

    # 难度输入分支
    diff_in = Input((1,), name="input_difficulty")
    d1 = Dense(16, activation="relu")(diff_in)

    # 单词ID输入分支
    wid_in = Input((1,), name="input_word_id", dtype="int32")
    wemb = Flatten()(Embedding(vocab_size, EMBEDDING_DIM)(wid_in))

    # 用户偏置输入分支
    bias_in = Input((1,), name="input_user_bias")
    b1 = Dense(16, activation="relu")(bias_in)


    # 合并所有特征
    # x: 历史序列 Transformer 输出
    # g: 猜词序列 Transformer 输出
    z = Concatenate()([x, d1, wemb, b1, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(dropout_rate)(z)

    # 输出层
    out_steps = Dense(1, "linear", name="output_steps")(Dense(32, "relu")(z))
    out_succ = Dense(1, "sigmoid", name="output_success")(Dense(16, "relu")(z))

    # 更新模型输入列表
    model = Model([h_in, diff_in, wid_in, bias_in, guess_seq_in], [out_steps, out_succ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"output_steps": "mse",
              "output_success": "binary_crossentropy"},
        loss_weights={"output_steps": 1.0, "output_success": 0.5},
        metrics={"output_success": "accuracy"}
    )
    return model

# ==========================================================
# 评估函数 (与之前相同)
# ==========================================================
def evaluate_model(model, Xs):
    # X_grid 替换为 X_guess_seq
    X_seq, X_diff, X_wid, X_bias, X_guess_seq, y_steps, y_succ = Xs
    pred_steps, pred_prob = model.predict({
        "input_history": X_seq,
        "input_difficulty": X_diff,
        "input_word_id": X_wid,
        "input_user_bias": X_bias,
        "input_guess_sequence": X_guess_seq # <-- 替换为新的输入键
    }, batch_size=1024, verbose=1)
    pred_steps = pred_steps.flatten()
    pred_prob = pred_prob.flatten()

    mae = mean_absolute_error(y_steps, np.clip(pred_steps, 0, 7))
    rmse = np.sqrt(mean_squared_error(y_steps, np.clip(pred_steps, 0, 7)))
    acc = accuracy_score(y_succ.astype(int), (pred_prob >= 0.5).astype(int))
    try:
        auc = roc_auc_score(y_succ, pred_prob)
    except:
        auc = float("nan")

    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}, AUC={auc}")
    return mae, rmse, acc, auc

def compute_large_error_rate(y_true, y_pred, threshold):
    errors = np.abs(y_true - y_pred)
    return np.mean(errors > threshold)

def plot_roc_curve(y_true, y_pred, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred):.3f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"AUC curve saved to: {save_path}")

def plot_loss(history, save_path):
    plt.figure(figsize=(12, 6))
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot component losses
    plt.subplot(1, 2, 2)
    if 'output_steps_loss' in history.history:
        plt.plot(history.history['output_steps_loss'], label='Training Steps Loss')
        if 'val_output_steps_loss' in history.history:
            plt.plot(history.history['val_output_steps_loss'], label='Validation Steps Loss')
    if 'output_success_loss' in history.history:
        plt.plot(history.history['output_success_loss'], label='Training Success Loss')
        if 'val_output_success_loss' in history.history:
            plt.plot(history.history['val_output_success_loss'], label='Validation Success Loss')
    plt.title('Component Losses')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Loss curve saved to: {save_path}")

# ==========================================================
# WandB-safe Keras Callback
# ==========================================================
class WandbEpochLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        metrics = {k: float(v) for k, v in logs.items()}
        metrics["epoch"] = int(epoch)
        wandb.log(metrics, step=epoch)

# ==========================================================
# 主程序
# ==========================================================
def main_train():
    set_seed(SEED)
    ensure_dirs()

    # WandB 初始化
    wandb.init(
        project="word-difficulty-prediction",
        name="transformer-guess-sequence-run", # 更新运行名称
        config={
            "model_type": "Transformer (Guess Seq)",
            "look_back": LOOK_BACK,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "project_dim": PROJECT_DIM,
            "num_heads": NUM_HEADS,
            "ff_dim": FF_DIM,
            "transformer_layers": TRANSFORMER_LAYERS,
            "dropout_rate": DROPOUT_RATE,
            "embedding_dim": EMBEDDING_DIM,
            "seed": SEED,
            "guess_sequence_len": MAX_TRIES, # 更新参数
            "guess_feat_len": GRID_FEAT_LEN
        },
        settings=wandb.Settings(_disable_stats=True) 
    )

    try:
        if hasattr(wandb.run, "summary") and "graph" in wandb.run.summary:
            wandb.run.summary.pop("graph", None)
    except Exception:
        pass

    # 1. 数据读取
    use_cols_list = ["Game", "Trial", "Username", "target", "processed_text"]
    train_df = safe_read_csv(TRAIN_FILE, usecols=use_cols_list)
    val_df = safe_read_csv(VAL_FILE, usecols=use_cols_list)
    test_df = safe_read_csv(TEST_FILE, usecols=use_cols_list)

    # 2. 难度/用户水平
    diff_map = {}
    user_map = {}
    if os.path.exists(DIFFICULTY_FILE):
        ddf = pd.read_csv(DIFFICULTY_FILE)
        diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))

    # 3. Tokenizer（train-only）
    tokenizer = fit_tokenizer(train_df)

    # 4. 附加特征（含 guess sequence）
    train_df = attach_features(train_df, tokenizer, diff_map, user_map)
    val_df = attach_features(val_df, tokenizer, diff_map, user_map)
    test_df = attach_features(test_df, tokenizer, diff_map, user_map)

    # 5. Build histories
    hist_train = build_history(train_df)
    hist_val = build_history(val_df)
    hist_test = build_history(test_df)

    # 6. Sliding samples（包含 guess sequence）
    X_train = create_samples(hist_train, LOOK_BACK)
    X_val = create_samples(hist_val, LOOK_BACK)
    X_test = create_samples(hist_test, LOOK_BACK)

    print(f"Train={len(X_train[0])}, Val={len(X_val[0])}, Test={len(X_test[0])}")

    vocab_size = len(tokenizer.word_index) + 1

    # 7. Model
    model = build_model(LOOK_BACK, vocab_size)
    model.summary()

    # 8. TF dataset
    train_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_train[0],
            "input_difficulty": X_train[1],
            "input_word_id": X_train[2],
            "input_user_bias": X_train[3],
            "input_guess_sequence": X_train[4] # <-- 更新输入键
        },
        {
            "output_steps": X_train[5],
            "output_success": X_train[6]
        }
    )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_val[0],
            "input_difficulty": X_val[1],
            "input_word_id": X_val[2],
            "input_user_bias": X_val[3],
            "input_guess_sequence": X_val[4] # <-- 更新输入键
        },
        {
            "output_steps": X_val[5],
            "output_success": X_val[6]
        }
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 9. 训练
    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    wandb_logger = WandbEpochLogger()

    train_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=[early, wandb_logger]
    )

    # 绘制并保存损失曲线
    loss_curve_path = "visualization/Transformer_guess_seq_loss_curve.png"
    plot_loss(train_history, loss_curve_path)
    try:
        wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
    except Exception:
        pass

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 验证评估
    print("\n=== Validation ===")
    val_mae, val_rmse, val_acc, val_auc = evaluate_model(model, X_val)

    # 记录验证集指标到wandb
    wandb.log({"val_mae": val_mae, "val_rmse": val_rmse, "val_accuracy": val_acc, "val_auc": val_auc})

    # 绘制验证集AUC曲线
    val_pred_steps, val_pred_prob = model.predict({
        "input_history": X_val[0], "input_difficulty": X_val[1], "input_word_id": X_val[2],
        "input_user_bias": X_val[3], "input_guess_sequence": X_val[4] # <-- 更新输入键
    }, batch_size=1024, verbose=0)
    val_roc_curve_path = "visualization/Transformer_guess_seq_validation_roc_curve.png"
    plot_roc_curve(X_val[6], val_pred_prob.flatten(), val_roc_curve_path)
    try:
        wandb.log({"validation_roc_curve": wandb.Image(val_roc_curve_path)})
    except Exception:
        pass

    # 测试评估
    print("\n=== Test ===")
    test_mae, test_rmse, test_acc, test_auc = evaluate_model(model, X_test)

    wandb.log({"test_mae": test_mae, "test_rmse": test_rmse, "test_accuracy": test_acc, "test_auc": test_auc})

    # 绘制测试集AUC曲线
    test_pred_steps, test_pred_prob = model.predict({
        "input_history": X_test[0], "input_difficulty": X_test[1], "input_word_id": X_test[2],
        "input_user_bias": X_test[3], "input_guess_sequence": X_test[4] # <-- 更新输入键
    }, batch_size=1024, verbose=0)
    test_roc_curve_path = "visualization/Transformer_guess_seq_test_roc_curve.png"
    plot_roc_curve(X_test[6], test_pred_prob.flatten(), test_roc_curve_path)
    try:
        wandb.log({"test_roc_curve": wandb.Image(test_roc_curve_path)})
    except Exception:
        pass

    # --------------------------------------------------------
    # 生成大型误差统计
    # --------------------------------------------------------
    val_pred_steps = val_pred_steps.flatten()
    val_large_error_rate = compute_large_error_rate(X_val[5], np.clip(val_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    test_pred_steps = test_pred_steps.flatten()
    test_large_error_rate = compute_large_error_rate(X_test[5], np.clip(test_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    # --------------------------------------------------------
    # 格式化报告
    # --------------------------------------------------------
    report = f"""
========================================
 Transformer Guess Sequence Model Report 
========================================
---- Validation Set Metrics ----
1. Mean Absolute Error (MAE)    : {val_mae:.4f}
2. Root Mean Squared Error (RMSE)     : {val_rmse:.4f}
3. Win/Loss Prediction Accuracy        : {val_acc:.3%}
4. Area Under ROC Curve (AUC)   : {val_auc:.4f}
5. Large Error Rate (>{LARGE_ERROR_THRESHOLD} steps)  : {val_large_error_rate:.3%}

---- Test Set Metrics ----
1. Mean Absolute Error (MAE)    : {test_mae:.4f}
2. Root Mean Squared Error (RMSE)     : {test_rmse:.4f}
3. Win/Loss Prediction Accuracy        : {test_acc:.3%}
4. Area Under ROC Curve (AUC)   : {test_auc:.4f}
5. Large Error Rate (>{LARGE_ERROR_THRESHOLD} steps)  : {test_large_error_rate:.3%}
========================================
"""

    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 Report saved to: {REPORT_SAVE_PATH}")
    print(report)

    wandb.log({
        "val_large_error_rate": val_large_error_rate,
        "test_large_error_rate": test_large_error_rate
    })

    # 结束 wandb 运行
    wandb.finish()

# 预测模式（按需启用）
def main_predict(user_id):
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("请先训练模型。")

    model = tf.keras.models.load_model(MODEL_SAVE_PATH, custom_objects={'TransformerBlock': TransformerBlock})
    tokenizer = load_tokenizer()

    df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target", "processed_text"])
    diff_map = {}
    user_map = {}
    if os.path.exists(DIFFICULTY_FILE):
        ddf = pd.read_csv(DIFFICULTY_FILE)
        diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))

    # 附加 guess sequence 特征
    df = attach_features(df, tokenizer, diff_map, user_map)
    hist = build_history(df)

    if user_id not in hist:
        print(f"用户 {user_id} 无记录")
        return

    events = hist[user_id]
    if len(events) < 1:
        print("历史不足")
        return

    # 准备历史序列输入
    if len(events) < LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        # 填充的元组现在需要包含 (6, 8) 序列的占位符
        pad_guess_seq = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)
        pad = [(avg, 4.0, 0, 4.0, pad_guess_seq)] * (LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-LOOK_BACK:]

    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, LOOK_BACK, 2)

    last = events[-1] # 上一局游戏的数据

    # 目标事件的先验特征
    diff = np.array([[last[1] / 7.0]], np.float32)
    wid = np.array([[last[2]]], np.int32)
    bias = np.array([[last[3] / 7.0]], np.float32)
    
    # 使用上一局游戏结束时的猜词序列特征
    guess_seq = last[4].reshape(1, MAX_TRIES, GRID_FEAT_LEN) # Shape (1, 6, 8)

    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_difficulty": diff,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_guess_sequence": guess_seq # <-- 更新输入键
    }, verbose=0)

    print(f"预测步数: {float(np.clip(p_steps, 0, 6.99)):.2f}")
    print(f"成功概率: {float(p_prob):.3f}")

# ==========================================================
# 启动入口
# ==========================================================
if __name__ == "__main__":
    main_train()