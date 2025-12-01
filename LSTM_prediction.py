#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构版 LSTM 多输入预测脚本（含 Wordle grid 特征，WandB-safe 日志）
直接运行即开始训练（默认 RUN_MODE="train"）。
修正：将 parse_grid_column 中的填充逻辑改为使用空行，避免未来信息泄露。

新增：将玩家猜词过程编码为时间序列 (parse_grid_sequence)。
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
                                     LSTM, Bidirectional, Concatenate)
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
LEARNING_RATE = 0.0007

MODEL_SAVE_PATH = "models/lstm/lstm_model.keras"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.json"

# LSTM 架构参数
LSTM_UNITS = 64
DROPOUT_RATE = 0.3
EMBEDDING_DIM = 32

OOV_TOKEN = "<OOV>"

LARGE_ERROR_THRESHOLD = 1.5
PATIENCE = 4
REPORT_SAVE_PATH = "outputs/lstm_output.txt"

# 固定随机种子
SEED = 2009

# Wordle固定参数
MAX_TRIES = 6
GRID_FEAT_LEN = 8
# 新增: 序列特征长度 (每个时间步的特征数量)
GRID_SEQ_FEAT_DIM = 4 # 绿色、黄色、灰色计数 + 尝试次数归一化

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置确定性操作（可能对某些 TF 版本有影响）
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

def ensure_dirs():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH) or ".", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models/lstm", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing: {path}")
    return pd.read_csv(path, usecols=usecols)

# --------------------------
# Wordle grid parsing helper
# --------------------------
def parse_grid_column(grid_cell):
    """
    期望 grid_cell 类似 "['⬜⬜⬜⬜⬜','⬜⬜⬜🟨⬜',...]" 或已经是 list。
    返回长度为 GRID_FEAT_LEN 的浮点向量（统计特征）。
    若无法解析，返回全 0 向量。
    """
    if pd.isna(grid_cell):
        return np.zeros(GRID_FEAT_LEN, dtype=np.float32)

    # 解析网格列表
    if isinstance(grid_cell, (list, tuple)):
        grid_list = list(grid_cell)
    else:
        try:
            grid_list = ast.literal_eval(grid_cell)
            if not isinstance(grid_list, (list, tuple)):
                grid_list = [grid_list]
            grid_list = [str(r) for r in grid_list if isinstance(r, (str, bytes))]
        except Exception:
            return np.zeros(GRID_FEAT_LEN, dtype=np.float32)

    num_rows = len(grid_list)

    # 1. Padding 逻辑: 修正了使用最后一行填充的问题
    if num_rows < MAX_TRIES:
        # 用空行 "⬜⬜⬜⬜⬜" 填充未进行的尝试，而不是用最后一行重复填充，避免未来信息泄露。
        blank_row = "⬜⬜⬜⬜⬜" 
        padding_rows = [blank_row] * (MAX_TRIES - num_rows)
        padded_grid_list = grid_list + padding_rows
    elif num_rows > MAX_TRIES:
        # 如果超过 MAX_TRIES，则截断，只取前 MAX_TRIES 行（通常不应该发生）
        padded_grid_list = grid_list[:MAX_TRIES]
    else:
        # 恰好 MAX_TRIES 行或 0 行
        padded_grid_list = grid_list

    # 2. 统计特征 (基于 Padding 后的 6 行)
    greens = 0
    yellows = 0
    grays = 0
    pos_green_counts = np.zeros(5, dtype=np.float32)
    
    # 归一化基数
    norm_base_cells = float(MAX_TRIES * 5)  # 6 * 5 = 30
    norm_base_rows = float(MAX_TRIES)      # 6
    
    # 遍历 Padding 后的网格
    for row in padded_grid_list:
        if not isinstance(row, str) or len(row) != 5:
            continue
        for i, ch in enumerate(row):
            if ch == "🟩":
                greens += 1
                if i < 5:
                    pos_green_counts[i] += 1.0
            elif ch == "🟨":
                yellows += 1
            elif ch == "⬜" or ch == "⬛":
                grays += 1

    # 3. 构建特征向量
    feat = np.zeros(GRID_FEAT_LEN, dtype=np.float32)
    feat[0] = greens / norm_base_cells
    feat[1] = yellows / norm_base_cells
    feat[2] = grays / norm_base_cells
    
    # 位置绿占比：除以 MAX_TRIES (6)
    for i in range(5):
        feat[3 + i] = (pos_green_counts[i] / norm_base_rows)
        
    return feat


def parse_grid_sequence(grid_cell):
    """
    新增函数：将 grid 列表转换为一个时间序列特征矩阵。
    返回形状为 (MAX_TRIES, GRID_SEQ_FEAT_DIM) 的浮点矩阵。
    时间步 i 对应第 i 次尝试的结果。
    """
    if pd.isna(grid_cell):
        # 无法解析时返回全零序列
        return np.zeros((MAX_TRIES, GRID_SEQ_FEAT_DIM), dtype=np.float32)

    # 解析网格列表
    if isinstance(grid_cell, (list, tuple)):
        grid_list = list(grid_cell)
    else:
        try:
            grid_list = ast.literal_eval(grid_cell)
            if not isinstance(grid_list, (list, tuple)):
                grid_list = [grid_list]
            grid_list = [str(r) for r in grid_list if isinstance(r, (str, bytes))]
        except Exception:
            return np.zeros((MAX_TRIES, GRID_SEQ_FEAT_DIM), dtype=np.float32)

    num_rows = len(grid_list)
    seq_features = []
    
    # 1. 序列特征提取
    for t in range(MAX_TRIES):
        feat = np.zeros(GRID_SEQ_FEAT_DIM, dtype=np.float32)
        greens = 0
        yellows = 0
        grays = 0
        
        # 如果是有效的尝试
        if t < num_rows:
            row = grid_list[t]
            if isinstance(row, str) and len(row) == 5:
                for ch in row:
                    if ch == "🟩":
                        greens += 1
                    elif ch == "🟨":
                        yellows += 1
                    elif ch == "⬜" or ch == "⬛":
                        grays += 1
            
            # 特征 0-2: 颜色数量归一化 (除以 5)
            feat[0] = greens / 5.0
            feat[1] = yellows / 5.0
            feat[2] = grays / 5.0
            # 特征 3: 尝试次数归一化 (除以 6)
            feat[3] = (t + 1) / float(MAX_TRIES) 
        
        # 如果是未进行的尝试 (padding)，则特征向量为全 0，表示缺失信息
        
        seq_features.append(feat)

    return np.array(seq_features, dtype=np.float32)


# --------------------------
# Tokenizer
# --------------------------
def fit_tokenizer(train_df):
    tokenizer = Tokenizer(oov_token=OOV_TOKEN, filters='', lower=True)
    tokenizer.fit_on_texts(train_df["target"].astype(str))
    with open(TOKENIZER_PATH, "w", encoding="utf-8") as f:
        json.dump(tokenizer.word_index, f, indent=2)
    return tokenizer

def load_tokenizer():
    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    tk = Tokenizer(oov_token=OOV_TOKEN)
    tk.word_index = word_index
    return tk

# --------------------------
# 特征附加（包含 grid 统计和序列）
# --------------------------
def attach_features(df, tokenizer, diff_map, user_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    # 单词 id（只取第一个 token id 或 0）
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)

    # 解析 grid 列（如果存在）
    if "processed_text" in df.columns:
        df["grid_feat"] = df["processed_text"].apply(parse_grid_column)
        # 新增: 解析 grid 序列
        df["grid_seq"] = df["processed_text"].apply(parse_grid_sequence)
    else:
        df["grid_feat"] = [np.zeros(GRID_FEAT_LEN, dtype=np.float32) for _ in range(len(df))]
        # 新增: 缺失时返回零序列
        df["grid_seq"] = [np.zeros((MAX_TRIES, GRID_SEQ_FEAT_DIM), dtype=np.float32) for _ in range(len(df))]

    return df

# --------------------------
# 历史建表（每条记录存入 grid_feat 和 grid_seq）
# --------------------------
def build_history(df) -> Dict[str, List[Tuple]]:
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        hist[u] = [(int(r["Trial"]),
                    float(r["word_difficulty"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    np.array(r["grid_feat"], dtype=np.float32),  # 索引 4: 统计特征
                    np.array(r["grid_seq"], dtype=np.float32))   # 索引 5: 序列特征
                   for _, r in g.iterrows()]
    return hist

# --------------------------
# 滑窗生成样本（包含 grid 统计特征和序列特征）
# --------------------------
def create_samples(history, look_back):
    # X_grid 是统计特征, X_grid_seq 是序列特征
    X_seq, X_diff, X_wid, X_bias, X_grid, X_grid_seq, y_steps, y_succ = [], [], [], [], [], [], [], []
    for _, events in history.items():
        if len(events) <= look_back:
            continue
        for i in range(look_back, len(events)):
            window = events[i-look_back:i]
            target = events[i]

            trials = np.array([t[0] for t in window], np.float32)
            norm = trials / 7.0
            std = np.std(trials) / 7.0

            seq = np.stack([norm, np.full_like(norm, std)], axis=1)
            X_seq.append(seq)
            X_diff.append([target[1] / 7.0])
            X_wid.append([target[2]])
            X_bias.append([target[3] / 7.0])
            X_grid.append(target[4])      # 统计特征
            X_grid_seq.append(target[5])  # 序列特征

            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)

    if not X_seq:
        return (np.zeros((0, look_back, 2), np.float32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, 1), np.int32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, GRID_FEAT_LEN), np.float32),
                np.zeros((0, MAX_TRIES, GRID_SEQ_FEAT_DIM), np.float32), # 新增：序列特征形状
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32))

    return (
        np.array(X_seq, np.float32),
        np.array(X_diff, np.float32),
        np.array(X_wid, np.int32),
        np.array(X_bias, np.float32),
        np.array(X_grid, np.float32),
        np.array(X_grid_seq, np.float32), # 新增：序列特征数组
        np.array(y_steps, np.float32),
        np.array(y_succ, np.float32)
    )

# ==========================================================
# LSTM 模型（加入 grid 序列支持）
# ==========================================================
def build_model(look_back, vocab_size):
    # 历史输入分支 (玩家历史成绩序列)
    h_in = Input((look_back, 2), name="input_history")
    x = Bidirectional(LSTM(LSTM_UNITS, return_sequences=True))(h_in)
    x = Dropout(DROPOUT_RATE)(x)
    x = Bidirectional(LSTM(LSTM_UNITS // 2))(x)
    x = Dropout(DROPOUT_RATE)(x)

    # 难度输入分支
    diff_in = Input((1,), name="input_difficulty")
    d1 = Dense(16, activation="relu")(diff_in)

    # 单词ID输入分支
    wid_in = Input((1,), name="input_word_id", dtype="int32")
    wemb = Flatten()(Embedding(vocab_size, EMBEDDING_DIM)(wid_in))

    # 用户偏置输入分支
    bias_in = Input((1,), name="input_user_bias")
    b1 = Dense(16, activation="relu")(bias_in)

    # Wordle grid 统计输入分支
    grid_in = Input((GRID_FEAT_LEN,), name="input_grid_stat") # 改名以区分
    g1 = Dense(16, activation="relu")(grid_in)
    
    # 新增: Wordle grid 序列输入分支
    grid_seq_in = Input((MAX_TRIES, GRID_SEQ_FEAT_DIM), name="input_grid_sequence")
    g_seq = Bidirectional(LSTM(LSTM_UNITS // 4))(grid_seq_in)
    g_seq = Dropout(DROPOUT_RATE)(g_seq)
    g2 = Dense(16, activation="relu")(g_seq) # 降维

    # 合并所有特征
    # 注意: 增加了 g2 (grid_seq_in 的输出)
    z = Concatenate()([x, d1, wemb, b1, g1, g2]) 
    z = Dense(64, activation="relu")(z)
    z = Dropout(DROPOUT_RATE)(z)

    # 输出层
    out_steps = Dense(1, "linear", name="output_steps")(Dense(32, "relu")(z))
    out_succ = Dense(1, "sigmoid", name="output_success")(Dense(16, "relu")(z))

    # 更新模型输入列表
    model = Model([h_in, diff_in, wid_in, bias_in, grid_in, grid_seq_in], [out_steps, out_succ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
        loss={"output_steps": "mse",
              "output_success": "binary_crossentropy"},
        loss_weights={"output_steps": 1.0, "output_success": 0.5},
        metrics={"output_success": "accuracy"}
    )
    return model

# ==========================================================
# 评估函数
# ==========================================================
def evaluate_model(model, Xs):
    # Xs 索引更新: X_grid_seq 为索引 5
    X_seq, X_diff, X_wid, X_bias, X_grid, X_grid_seq, y_steps, y_succ = Xs
    pred_steps, pred_prob = model.predict({
        "input_history": X_seq,
        "input_difficulty": X_diff,
        "input_word_id": X_wid,
        "input_user_bias": X_bias,
        "input_grid_stat": X_grid, # 更新键名
        "input_grid_sequence": X_grid_seq # 新增输入
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
# WandB-safe Keras Callback（只记录 epoch 指标，不触发 graph 采样）
# ==========================================================
class WandbEpochLogger(Callback):
    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        # 将所有可记录的指标写入 wandb（带 step）
        # 使用 epoch 作为 step
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
        name="lstm-model-grid-seq-run",
        config={
            "model_type": "LSTM_Grid_Seq",
            "look_back": LOOK_BACK,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "embedding_dim": EMBEDDING_DIM,
            "seed": SEED,
            "grid_feat_len": GRID_FEAT_LEN,
            "grid_seq_feat_dim": GRID_SEQ_FEAT_DIM # 新增配置
        },
        settings=wandb.Settings(_disable_stats=True)  # 关闭某些自动统计，避免 graph 写入
    )

    # 尝试移除可能残留的 graph 字段（防御性）
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

    # 4. 附加特征（含 grid 统计和序列）
    train_df = attach_features(train_df, tokenizer, diff_map, user_map)
    val_df = attach_features(val_df, tokenizer, diff_map, user_map)
    test_df = attach_features(test_df, tokenizer, diff_map, user_map)

    # 5. Build histories
    hist_train = build_history(train_df)
    hist_val = build_history(val_df)
    hist_test = build_history(test_df)

    # 6. Sliding samples（包含 grid 统计和序列）
    # X_set 结构：(seq, diff, wid, bias, grid_stat, grid_seq, y_steps, y_succ)
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
            "input_grid_stat": X_train[4],      # 统计特征
            "input_grid_sequence": X_train[5]   # 序列特征
        },
        {
            "output_steps": X_train[6],
            "output_success": X_train[7]
        }
    )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_val[0],
            "input_difficulty": X_val[1],
            "input_word_id": X_val[2],
            "input_user_bias": X_val[3],
            "input_grid_stat": X_val[4],
            "input_grid_sequence": X_val[5]
        },
        {
            "output_steps": X_val[6],
            "output_success": X_val[7]
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
    loss_curve_path = "visualization/LSTM_loss_curve.png"
    plot_loss(train_history, loss_curve_path)

    # 将损失曲线上传到WandB
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
    wandb.log({
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_accuracy": val_acc,
        "val_auc": val_auc
    })

    # 绘制验证集AUC曲线 (使用更新后的索引 7: y_succ)
    val_pred_steps, val_pred_prob = model.predict({
        "input_history": X_val[0],
        "input_difficulty": X_val[1],
        "input_word_id": X_val[2],
        "input_user_bias": X_val[3],
        "input_grid_stat": X_val[4],
        "input_grid_sequence": X_val[5]
    }, batch_size=1024, verbose=0)
    val_roc_curve_path = "visualization/LSTM_validation_roc_curve.png"
    plot_roc_curve(X_val[7], val_pred_prob.flatten(), val_roc_curve_path)
    try:
        wandb.log({"validation_roc_curve": wandb.Image(val_roc_curve_path)})
    except Exception:
        pass

    # 测试评估
    print("\n=== Test ===")
    test_mae, test_rmse, test_acc, test_auc = evaluate_model(model, X_test)

    wandb.log({
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_accuracy": test_acc,
        "test_auc": test_auc
    })

    # 绘制测试集AUC曲线 (使用更新后的索引 7: y_succ)
    test_pred_steps, test_pred_prob = model.predict({
        "input_history": X_test[0],
        "input_difficulty": X_test[1],
        "input_word_id": X_test[2],
        "input_user_bias": X_test[3],
        "input_grid_stat": X_test[4],
        "input_grid_sequence": X_test[5]
    }, batch_size=1024, verbose=0)
    test_roc_curve_path = "visualization/LSTM_test_roc_curve.png"
    plot_roc_curve(X_test[7], test_pred_prob.flatten(), test_roc_curve_path)
    try:
        wandb.log({"test_roc_curve": wandb.Image(test_roc_curve_path)})
    except Exception:
        pass

    # --------------------------------------------------------
    # 生成大型误差统计 (使用更新后的索引 6: y_steps)
    # --------------------------------------------------------
    val_pred_steps, _ = model.predict({
        "input_history": X_val[0],
        "input_difficulty": X_val[1],
        "input_word_id": X_val[2],
        "input_user_bias": X_val[3],
        "input_grid_stat": X_val[4],
        "input_grid_sequence": X_val[5]
    }, batch_size=1024, verbose=0)
    val_pred_steps = val_pred_steps.flatten()
    val_large_error_rate = compute_large_error_rate(X_val[6], np.clip(val_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    test_pred_steps, _ = model.predict({
        "input_history": X_test[0],
        "input_difficulty": X_test[1],
        "input_word_id": X_test[2],
        "input_user_bias": X_test[3],
        "input_grid_stat": X_test[4],
        "input_grid_sequence": X_test[5]
    }, batch_size=1024, verbose=0)
    test_pred_steps = test_pred_steps.flatten()
    test_large_error_rate = compute_large_error_rate(X_test[6], np.clip(test_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    # --------------------------------------------------------
    # 格式化报告
    # --------------------------------------------------------
    report = f"""
========================================
 LSTM Model Validation and Test Report 
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

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
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

    df = attach_features(df, tokenizer, diff_map, user_map)
    hist = build_history(df)

    if user_id not in hist:
        print(f"用户 {user_id} 无记录")
        return

    events = hist[user_id]
    if len(events) < 1:
        print("历史不足")
        return

    # 准备输入
    if len(events) < LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        # 填充 tuple 长度需要匹配 build_history 中的 6 个元素
        pad_event = (avg, 4.0, 0, 4.0, np.zeros(GRID_FEAT_LEN, dtype=np.float32), 
                     np.zeros((MAX_TRIES, GRID_SEQ_FEAT_DIM), dtype=np.float32))
        pad = [pad_event] * (LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-LOOK_BACK:]

    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, LOOK_BACK, 2)

    last = events[-1]
    diff = np.array([[last[1] / 7.0]], np.float32)
    wid = np.array([[last[2]]], np.int32)
    bias = np.array([[last[3] / 7.0]], np.float32)
    grid_stat = last[4].reshape(1, GRID_FEAT_LEN) # 统计特征 (索引 4)
    grid_seq = last[5].reshape(1, MAX_TRIES, GRID_SEQ_FEAT_DIM) # 序列特征 (索引 5)

    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_difficulty": diff,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_grid_stat": grid_stat,
        "input_grid_sequence": grid_seq
    }, verbose=0)

    print(f"预测步数: {float(np.clip(p_steps, 0, 6.99)):.2f}")
    print(f"成功概率: {float(p_prob):.3f}")

# ==========================================================
# 启动入口
# ==========================================================
if __name__ == "__main__":
    main_train()
    
# < 