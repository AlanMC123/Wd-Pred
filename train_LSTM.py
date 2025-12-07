
"""
LSTM 多输入预测脚本
直接运行即开始训练。
利用早停、Dropout、L2正则化防止过拟合。

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
                                     LSTM, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, Callback
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from predict import plot_roc_curve, plot_scatter
from tensorflow.keras.regularizers import l2 


# ==========================================================
# 全局配置
# ==========================================================

# 数据集和特征文件路径
TRAIN_FILE = "dataset/train_data.csv"
VAL_FILE = "dataset/val_data.csv"
TEST_FILE = "dataset/test_data.csv"
PLAYER_FILE = "dataset/player_data.csv" 
DIFFICULTY_FILE = "dataset/difficulty_data.csv"

# 模型和报告输出路径
MODEL_SAVE_PATH = "models/lstm/lstm_model.keras"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.json"
REPORT_SAVE_PATH = "outputs/lstm_output.txt"

# 训练基本参数
LOOK_BACK = 5
BATCH_SIZE = 1024
EPOCHS = 40
LEARNING_RATE = 0.0005
LARGE_ERROR_THRESHOLD = 1.5

# LSTM 架构参数
LSTM_UNITS = 56
DROPOUT_RATE = 0.45 
EMBEDDING_DIM = 24

# 词典参数
OOV_TOKEN = "<OOV>"

# 早停值
PATIENCE = 4

# 损失函数权重
LOSS_WEIGHTS = {"output_steps": 0.8, "output_success": 1}

# Focal Loss 超参数
FOCAL_LOSS_ALPHA = 0.25
FOCAL_LOSS_GAMMA = 2.0

# L2 正则化系数
L2_REG_FACTOR = 0.001 

# 固定随机种子
SEED = 42

# Wordle固定参数
MAX_TRIES = 6
GRID_FEAT_LEN = 8 # 3个累积颜色特征 + 5个位置绿色特征

# --------------------------
# 基本函数
# --------------------------
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
    os.makedirs("models/lstm", exist_ok=True)
    os.makedirs("visualization", exist_ok=True)

def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing: {path}")
    return pd.read_csv(path, usecols=usecols)

# --------------------------
# 网格序列解析器
# --------------------------
def encode_guess_sequence(grid_cell):
    """
    将 grid 列表转换为一个时间序列特征矩阵。
    返回形状为 (MAX_TRIES, GRID_FEAT_LEN) 的浮点矩阵。
    与Transformer模型使用相同的8维累积特征：
    1. 累积绿色方块数（归一化）
    2. 累积黄色方块数（归一化）
    3. 累积灰色方块数（归一化）
    4-8. 每个位置累积绿色方块数（归一化）
    """
    default_seq = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)
    if pd.isna(grid_cell):
        return default_seq
    try:
        if isinstance(grid_cell, (list, tuple)):
            grid_list = list(grid_cell)
        else:
            grid_list = ast.literal_eval(grid_cell)
            if not isinstance(grid_list, (list, tuple)):
                grid_list = [grid_list]
            grid_list = [str(r) for r in grid_list if isinstance(r, (str, bytes))]
    except Exception:
        return default_seq

    num_rows = len(grid_list)
    cumulative_greens = 0
    cumulative_yellows = 0
    cumulative_grays = 0
    cumulative_pos_green_counts = np.zeros(5, dtype=np.float32)
    feature_sequence = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)
    norm_base_cells = float(MAX_TRIES * 5)
    norm_base_rows = float(MAX_TRIES)

    for t in range(MAX_TRIES):
        if t < num_rows:
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
            cumulative_greens += greens_t
            cumulative_yellows += yellows_t
            cumulative_grays += grays_t
            cumulative_pos_green_counts += pos_green_counts_t

        feat = np.zeros(GRID_FEAT_LEN, dtype=np.float32)
        feat[0] = cumulative_greens / norm_base_cells
        feat[1] = cumulative_yellows / norm_base_cells
        feat[2] = cumulative_grays / norm_base_cells
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
# 单词难度、用户偏置特征附加
# --------------------------
def attach_features(df, tokenizer, user_map, diff_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    # 单词 id
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    # 添加单词难度
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)
    # 解析 grid 序列
    if "processed_text" in df.columns:
        df["grid_seq"] = df["processed_text"].apply(encode_guess_sequence)
    else:
        # 缺失时返回零序列
        df["grid_seq"] = [np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32) for _ in range(len(df))]

    return df

# ==========================================================
# 损失函数 (Focal Loss 定义)
# ==========================================================

def focal_loss(gamma=2.0, alpha=0.25):
    """
    Focal Loss for Binary Classification (sigmoid output).
    Reference: Lin et al., 2017.
    """
    gamma = float(gamma)
    alpha = float(alpha)

    def focal_loss_fixed(y_true, y_pred):
        # 裁剪 y_pred 以避免 log(0)
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)

        # 计算交叉熵
        bce = y_true * tf.math.log(y_pred)
        bce += (1 - y_true) * tf.math.log(1 - y_pred)
        bce = -bce

        # 计算调制因子
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)

        # 乘以权重项
        alpha_factor = y_true * alpha + (1 - y_true) * (1.0 - alpha)

        # Focal Loss = alpha_factor * modulating_factor * BCE
        focal_loss = alpha_factor * modulating_factor * bce
        
        return tf.reduce_mean(focal_loss)

    focal_loss_fixed.__name__ = f'focal_loss(gamma={gamma},alpha={alpha})'
    return focal_loss_fixed


# --------------------------
# 历史建表
# --------------------------
def build_history(df) -> Dict[str, List[Tuple]]:
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        # 历史记录 tuple 结构：(Trial, word_id, user_bias, word_difficulty, grid_seq)
        hist[u] = [(int(r["Trial"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    float(r["word_difficulty"]), 
                    np.array(r["grid_seq"], dtype=np.float32))
                   for _, r in g.iterrows()]
    return hist

# --------------------------
# 滑窗生成样本
# --------------------------
def create_samples(history, look_back):
    X_seq, X_wid, X_bias, X_diff, X_grid_seq, y_steps, y_succ = [], [], [], [], [], [], []
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
            # 单词 ID
            X_wid.append([target[1]])
            # 用户偏置
            X_bias.append([target[2] / 7.0])
            # 单词难度
            X_diff.append([target[3] / 7.0])
            # 序列特征
            X_grid_seq.append(target[4]) 

            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)

    if not X_seq:
        return (np.zeros((0, look_back, 2), np.float32),
                np.zeros((0, 1), np.int32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, MAX_TRIES, GRID_FEAT_LEN), np.float32), 
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32))

    return (
        np.array(X_seq, np.float32),
        np.array(X_wid, np.int32),
        np.array(X_bias, np.float32),
        np.array(X_diff, np.float32),
        np.array(X_grid_seq, np.float32),
        np.array(y_steps, np.float32),
        np.array(y_succ, np.float32)
    )

# ==========================================================
# LSTM 模型构建
# ==========================================================
def build_model(look_back, vocab_size):
    # 历史输入分支
    h_in = Input((look_back, 2), name="input_history")
    x = LSTM(LSTM_UNITS, kernel_regularizer=l2(L2_REG_FACTOR))(h_in)
    x = Dropout(DROPOUT_RATE)(x)

    # 单词 ID
    wid_in = Input((1,), name="input_word_id", dtype="int32")
    wemb = Flatten()(Embedding(vocab_size, EMBEDDING_DIM)(wid_in))

    # 用户偏置
    bias_in = Input((1,), name="input_user_bias")
    b1 = Dense(16, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(bias_in)

    # 单词难度
    diff_in = Input((1,), name="input_difficulty")
    d1 = Dense(16, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(diff_in)

    # Wordle 序列特征
    grid_seq_in = Input((MAX_TRIES, GRID_FEAT_LEN), name="input_grid_sequence")
    g_seq = LSTM(LSTM_UNITS // 4, kernel_regularizer=l2(L2_REG_FACTOR))(grid_seq_in)
    g_seq = Dropout(DROPOUT_RATE)(g_seq)
    g2 = Dense(16, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(g_seq)

    # 合并特征
    z = Concatenate()([x, wemb, b1, d1, g2])
    z = Dense(64, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(z)
    z = Dropout(DROPOUT_RATE)(z)

    # 回归头（预测步数）
    out_steps = Dense(1, "linear", name="output_steps")(Dense(32, "relu", kernel_regularizer=l2(L2_REG_FACTOR))(z))

    # success head
    succ = Dense(64, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(z)
    succ = Dropout(0.45)(succ) # <--- 關鍵修改：從 0.3 調整為 0.45
    succ = Dense(32, activation="relu", kernel_regularizer=l2(L2_REG_FACTOR))(succ)
    out_succ = Dense(1, activation="sigmoid", name="output_success")(succ)

    # 编译
    model = Model(
        [h_in, wid_in, bias_in, diff_in, grid_seq_in],
        [out_steps, out_succ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),

        # 使用 Focal Loss
        loss={
            "output_steps": "mae",
            "output_success": focal_loss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)
        },
        loss_weights=LOSS_WEIGHTS,
        metrics={"output_success": "accuracy"}
    )

    return model

# ==========================================================
# 评估函数
# ==========================================================
def evaluate_model(model, Xs):
    # Xs 结构: (seq, wid, bias, diff, grid_seq, y_steps, y_succ)
    X_seq, X_wid, X_bias, X_diff, X_grid_seq, y_steps, y_succ = Xs

    pred_steps, pred_prob = model.predict({
        "input_history": X_seq,
        "input_word_id": X_wid,
        "input_user_bias": X_bias,
        "input_difficulty": X_diff,
        "input_grid_sequence": X_grid_seq
    }, batch_size=1024, verbose=1)

    pred_steps = pred_steps.flatten()
    pred_prob = pred_prob.flatten()

    mae = mean_absolute_error(y_steps, np.clip(pred_steps, 0, 7))
    rmse = np.sqrt(mean_squared_error(y_steps, np.clip(pred_steps, 0, 7)))
    acc = accuracy_score(y_succ.astype(int), (pred_prob >= 0.5).astype(int))

    # 使用从predict.py导入的calculate_auc_best函数
    from predict import calculate_auc_best
    auc_value, _, _ = calculate_auc_best(y_succ, pred_prob)
    auc = auc_value

    print(f"MAE={mae:.4f}, RMSE={rmse:.4f}, ACC={acc:.4f}, AUC={auc:.4f}")
    return mae, rmse, acc, auc


def compute_large_error_rate(y_true, y_pred, threshold):
    errors = np.abs(y_true - y_pred)
    return np.mean(errors > threshold)

def plot_loss(history, save_path_base):
    # 确保保存路径是文件夹，以便保存多个文件
    save_dir = os.path.dirname(save_path_base) or "."
    
    # -------------------
    # 图 1: Training and Validation Loss (总损失 - 保持不变)
    # -------------------
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['loss'], label='Training Total Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Total Loss')
    plt.title('Training and Validation Total Loss (Weighted)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    total_loss_path = os.path.join(save_dir, "LSTM_total_loss_curve.png")
    plt.savefig(total_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Total Loss curve saved to: {total_loss_path}")
    
    # -------------------
    # 图 2: Steps Loss (回归任务 - MAE)
    # -------------------
    plt.figure(figsize=(6, 6))
    if 'output_steps_loss' in history.history:
        plt.plot(history.history['output_steps_loss'], label='Training Steps Loss')
        if 'val_output_steps_loss' in history.history:
            plt.plot(history.history['val_output_steps_loss'], label='Validation Steps Loss')
    plt.title('Steps Prediction Component Loss (MAE)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (MAE)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    steps_loss_path = os.path.join(save_dir, "LSTM_steps_loss_curve.png")
    plt.savefig(steps_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Steps Loss curve saved to: {steps_loss_path}")

    # -------------------
    # 图 3: Success Loss (分类任务 - Focal Loss)
    # -------------------
    plt.figure(figsize=(6, 6))
    if 'output_success_loss' in history.history:
        plt.plot(history.history['output_success_loss'], label='Training Success Loss')
        if 'val_output_success_loss' in history.history:
            plt.plot(history.history['val_output_success_loss'], label='Validation Success Loss')
    plt.title('Success Prediction Component Loss (Focal Loss)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (Focal Loss)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    success_loss_path = os.path.join(save_dir, "LSTM_success_loss_curve.png")
    plt.savefig(success_loss_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Success Loss curve saved to: {success_loss_path}")

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
        name="lstm-model-grid-seq-run",
        config={
            "model_type": "LSTM_Grid_Seq_Simplified",
            "look_back": LOOK_BACK,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "embedding_dim": EMBEDDING_DIM,
            "seed": SEED,
            "GRID_FEAT_LEN": GRID_FEAT_LEN # 新增配置
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
    user_map = {}
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
    
    # 加载单词难度数据
    diff_map = {}
    if os.path.exists(DIFFICULTY_FILE):
        df_diff = pd.read_csv(DIFFICULTY_FILE)
        # 使用平均尝试次数作为难度值
        diff_map = dict(zip(df_diff["word"], df_diff["avg_trial"]))

    # 3. Tokenizer
    tokenizer = fit_tokenizer(train_df)

    # 4. 附加特征
    train_df = attach_features(train_df, tokenizer, user_map, diff_map)
    val_df = attach_features(val_df, tokenizer, user_map, diff_map)
    test_df = attach_features(test_df, tokenizer, user_map, diff_map)

    # 5. 构建用户历史行为序列
    hist_train = build_history(train_df)
    hist_val = build_history(val_df)
    hist_test = build_history(test_df)

    # 6. Sliding samples
    # X_set 结构：(seq, wid, bias, grid_seq, y_steps, y_succ)
    X_train = create_samples(hist_train, LOOK_BACK)
    X_val = create_samples(hist_val, LOOK_BACK)
    X_test = create_samples(hist_test, LOOK_BACK)

    print(f"Train={len(X_train[0])}, Val={len(X_val[0])}, Test={len(X_test[0])}")

    vocab_size = len(tokenizer.word_index) + 1

    # 7. 模型构建
    model = build_model(LOOK_BACK, vocab_size)
    model.summary()

    # 8. TF dataset
    train_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_train[0],
            "input_word_id": X_train[1],
            "input_user_bias": X_train[2],
            "input_difficulty": X_train[3],
            "input_grid_sequence": X_train[4]
        },
        {
            "output_steps": X_train[5],
            "output_success": X_train[6]
        }
    )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_val[0],
            "input_word_id": X_val[1],
            "input_user_bias": X_val[2],
            "input_difficulty": X_val[3],
            "input_grid_sequence": X_val[4]
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
    loss_curve_path = "visualization/LSTM_loss_curve.png"
    plot_loss(train_history, loss_curve_path)

    try:
        wandb.log({"loss_curve": wandb.Image(loss_curve_path)})
    except Exception:
        pass

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # 验证评估
    print("\n=== Validation ===")
    # X_val 结构: (seq, wid, bias, diff, grid_seq, y_steps, y_succ)
    val_mae, val_rmse, val_acc, val_auc = evaluate_model(model, X_val)

    # 记录验证集指标到wandb
    wandb.log({
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_accuracy": val_acc,
        "val_auc": val_auc
    })

    # 绘制验证集ROC曲线
    val_pred_steps, val_pred_prob = model.predict({
        "input_history": X_val[0],
        "input_word_id": X_val[1],
        "input_user_bias": X_val[2],
        "input_difficulty": X_val[3],
        "input_grid_sequence": X_val[4]
    }, batch_size=1024, verbose=0)
    val_roc_curve_path = "visualization/LSTM_validation_roc_curve.png"
    plot_roc_curve(X_val[6], val_pred_prob.flatten(), val_roc_curve_path)
    
    # 绘制验证集散点图
    val_scatter_path = "visualization/LSTM_validation_scatter.png"
    plot_scatter(X_val[5], np.clip(val_pred_steps.flatten(), 0, 7), val_scatter_path, model_name="LSTM")
    
    try:
        wandb.log({"validation_roc_curve": wandb.Image(val_roc_curve_path), "validation_scatter": wandb.Image(val_scatter_path)})
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

    # 绘制测试集ROC曲线
    test_pred_steps, test_pred_prob = model.predict({
        "input_history": X_test[0],
        "input_word_id": X_test[1],
        "input_user_bias": X_test[2],
        "input_difficulty": X_test[3],
        "input_grid_sequence": X_test[4]
    }, batch_size=1024, verbose=0)
    test_roc_curve_path = "visualization/LSTM_test_roc_curve.png"
    plot_roc_curve(X_test[6], test_pred_prob.flatten(), test_roc_curve_path)
    
    # 绘制测试集散点图
    test_scatter_path = "visualization/LSTM_test_scatter.png"
    plot_scatter(X_test[5], np.clip(test_pred_steps.flatten(), 0, 7), test_scatter_path, model_name="LSTM")
    
    try:
        wandb.log({"test_roc_curve": wandb.Image(test_roc_curve_path), "test_scatter": wandb.Image(test_scatter_path)})
    except Exception:
        pass

    # 生成大型误差统计
    val_pred_steps, _ = model.predict({
        "input_history": X_val[0],
        "input_word_id": X_val[1],
        "input_user_bias": X_val[2],
        "input_difficulty": X_val[3],
        "input_grid_sequence": X_val[4]
    }, batch_size=1024, verbose=0)
    val_pred_steps = val_pred_steps.flatten()
    val_large_error_rate = compute_large_error_rate(X_val[5], np.clip(val_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    test_pred_steps, _ = model.predict({
        "input_history": X_test[0],
        "input_word_id": X_test[1],
        "input_user_bias": X_test[2],
        "input_difficulty": X_test[3],
        "input_grid_sequence": X_test[4]
    }, batch_size=1024, verbose=0)
    test_pred_steps = test_pred_steps.flatten()
    test_large_error_rate = compute_large_error_rate(X_test[5], np.clip(test_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    # 格式化报告
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

# ==========================================================
# 启动入口
# ==========================================================
if __name__ == "__main__":
    main_train()