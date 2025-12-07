"""
Transformer 多输入预测脚本 - "保底准确率"激进策略版
直接运行即开始训练。

核心修改：
- [Threshold Strategy] 采用 "Accuracy Constrained Recall Maximization" 策略。
  逻辑：在保证总体准确率 >= 85% (MIN_ACCURACY) 的前提下，寻找负类召回率最高的阈值。
- [Loss] 保持 Focal Loss 以优化训练目标。
- [Metrics] 包含 RMSE 和 负类精/召率。
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
from sklearn.metrics import (mean_absolute_error, mean_squared_error, 
                             accuracy_score, precision_recall_curve, f1_score, 
                             recall_score, precision_score, fbeta_score)
from predict import plot_roc_curve, plot_scatter

# ==========================================================
# 全局配置
# ==========================================================

# 核心参数：总体准确率的底线
# 只要准确率高于这个数，我们就尽可能提高阈值去抓失败局
MIN_ACCEPTABLE_ACCURACY = 0.85 

# 数据集和特征文件路径
TRAIN_FILE = "dataset/train_data.csv"
VAL_FILE = "dataset/val_data.csv"
TEST_FILE = "dataset/test_data.csv"
DIFFICULTY_FILE = "dataset/difficulty.csv"
PLAYER_FILE = "dataset/player_data.csv"

# 模型和报告输出路径
MODEL_SAVE_PATH = "models/transformer/transformer_model.keras"
TOKENIZER_PATH = "models/transformer/transformer_tokenizer.json"
REPORT_SAVE_PATH = "outputs/transformer_output.txt"

# 训练基本参数
LOOK_BACK = 5
BATCH_SIZE = 1024
EPOCHS = 30
LEARNING_RATE = 0.0002
LARGE_ERROR_THRESHOLD = 1.5

# Transformer 架构参数
PROJECT_DIM = 16
NUM_HEADS = 6
FF_DIM = 16
TRANSFORMER_LAYERS = 1
DROPOUT_RATE = 0.45
EMBEDDING_DIM = 16

# 词典参数
OOV_TOKEN = "<OOV>"

# 早停值
PATIENCE = 3

# 损失函数权重
LOSS_WEIGHTS = {"output_steps": 0.2, "output_success": 1.0}

# Focal Loss 超参数
FOCAL_LOSS_ALPHA = 0.25 
FOCAL_LOSS_GAMMA = 2.0  

# 固定随机种子
SEED = 42

# Wordle固定参数
MAX_TRIES = 6
GRID_FEAT_LEN = 8 

# ==========================================================
# 基本函数
# ==========================================================
def set_seed(seed):
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
# 网格序列解析器
# --------------------------
def encode_guess_sequence(grid_cell):
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
# Tokenizer & Features
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

def attach_features(df, tokenizer, user_map, diff_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    if "processed_text" in df.columns:
        df["guess_seq_feat"] = df["processed_text"].apply(encode_guess_sequence)
    else:
        df["guess_seq_feat"] = [np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32) for _ in range(len(df))]
    return df

def build_history(df) -> Dict[str, List[Tuple]]:
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        hist[u] = [(int(r["Trial"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    float(r["word_difficulty"]),
                    np.array(r["guess_seq_feat"], dtype=np.float32))
                   for _, r in g.iterrows()]
    return hist

def create_samples(history, look_back):
    X_seq, X_wid, X_bias, X_diff, X_guess_seq, y_steps, y_succ = [], [], [], [], [], [], []
    for user, events in history.items():
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
            X_wid.append([target[1]])
            X_bias.append([target[2] / 7.0])
            X_diff.append([target[3] / 7.0])
            X_guess_seq.append(window[-1][4])
            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)

    if not X_seq:
        return (np.zeros((0, look_back, 2), np.float32), np.zeros((0, 1), np.int32), np.zeros((0, 1), np.float32), np.zeros((0, 1), np.float32), np.zeros((0, MAX_TRIES, GRID_FEAT_LEN), np.float32), np.zeros((0,), np.float32), np.zeros((0,), np.float32))

    return (np.array(X_seq, np.float32), np.array(X_wid, np.int32), np.array(X_bias, np.float32), np.array(X_diff, np.float32), np.array(X_guess_seq, np.float32), np.array(y_steps, np.float32), np.array(y_succ, np.float32))

# ==========================================================
# Focal Loss
# ==========================================================
def focal_loss(gamma=2.0, alpha=0.25):
    gamma = float(gamma)
    alpha = float(alpha)
    def focal_loss_fixed(y_true, y_pred):
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        bce = y_true * tf.math.log(y_pred)
        bce += (1 - y_true) * tf.math.log(1 - y_pred)
        bce = -bce
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        modulating_factor = tf.pow(1.0 - p_t, gamma)
        alpha_factor = y_true * alpha + (1 - y_true) * (1.0 - alpha)
        loss = alpha_factor * modulating_factor * bce
        return tf.reduce_mean(loss)
    focal_loss_fixed.__name__ = f'focal_loss(gamma={gamma},alpha={alpha})'
    return focal_loss_fixed

# ==========================================================
# Transformer Blocks
# ==========================================================
class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        super(TransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.rate = rate
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([Dense(ff_dim, activation="relu"), Dense(embed_dim),])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)
    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
    def get_config(self):
        config = super(TransformerBlock, self).get_config()
        config.update({'embed_dim': self.embed_dim, 'num_heads': self.num_heads, 'ff_dim': self.ff_dim, 'rate': self.rate})
        return config
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_model(look_back, vocab_size, project_dim=PROJECT_DIM, num_heads=NUM_HEADS, ff_dim=FF_DIM, n_layers=TRANSFORMER_LAYERS, dropout_rate=DROPOUT_RATE):
    h_in = Input((look_back, 2), name="input_history")
    x = Dense(project_dim, activation="relu")(h_in)
    positions = tf.range(start=0, limit=look_back, delta=1)
    pos_emb = Embedding(input_dim=look_back, output_dim=project_dim)(positions)
    x = x + pos_emb
    for _ in range(n_layers):
        x = TransformerBlock(project_dim, num_heads, ff_dim, dropout_rate)(x)
    x = GlobalAveragePooling1D()(x)
    x = Dropout(dropout_rate)(x)

    guess_seq_in = Input((MAX_TRIES, GRID_FEAT_LEN), name="input_guess_sequence")
    g = Dense(project_dim, activation="relu")(guess_seq_in)
    positions_g = tf.range(start=0, limit=MAX_TRIES, delta=1)
    pos_emb_g = Embedding(input_dim=MAX_TRIES, output_dim=project_dim)(positions_g)
    g = g + pos_emb_g
    for _ in range(n_layers):
        g = TransformerBlock(project_dim, num_heads, ff_dim, dropout_rate)(g)
    g = GlobalAveragePooling1D()(g)
    g = Dropout(dropout_rate)(g)

    wid_in = Input((1,), name="input_word_id", dtype="int32")
    wemb = Flatten()(Embedding(vocab_size, EMBEDDING_DIM)(wid_in))
    bias_in = Input((1,), name="input_user_bias")
    b1 = Dense(16, activation="relu")(bias_in)
    diff_in = Input((1,), name="input_difficulty")
    d1 = Dense(16, activation="relu")(diff_in)

    z = Concatenate()([x, wemb, b1, d1, g])
    z = Dense(64, activation="relu")(z)
    z = Dropout(dropout_rate)(z)

    steps = Dense(32, "relu")(z)
    steps = Dropout(0.2)(steps)
    out_steps = Dense(1, "linear", name="output_steps")(steps)
    
    succ = Dense(64, activation="relu")(z)
    succ = Dropout(0.3)(succ)
    succ = Dense(32, activation="relu")(succ)
    succ = Dropout(0.2)(succ)
    out_succ = Dense(1, activation="sigmoid", name="output_success")(succ)
    
    model = Model([h_in, wid_in, bias_in, diff_in, guess_seq_in], [out_steps, out_succ])
    model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE), loss={"output_steps": "mse", "output_success": focal_loss(alpha=FOCAL_LOSS_ALPHA, gamma=FOCAL_LOSS_GAMMA)}, loss_weights = LOSS_WEIGHTS, metrics={"output_success": "accuracy"})
    return model

# ==========================================================
# 激进阈值寻优 (Aggressive Threshold Strategy)
# ==========================================================

def find_optimal_threshold(y_true, y_prob):
    """
    [保底激进版] 
    策略：寻找能够最大化负类召回率(抓出更多输局)的阈值，
    但在该阈值下，总体准确率(Accuracy)不能低于 MIN_ACCEPTABLE_ACCURACY (如 0.85)。
    """
    min_acc = MIN_ACCEPTABLE_ACCURACY
    
    # 搜索范围：0.50 到 0.99
    thresholds = np.arange(0.5, 0.99, 0.005)
    
    print(f"   > Searching threshold: Max Negative Recall s.t. Accuracy >= {min_acc:.1%}...")
    print(f"     (Threshold | Acc | Neg_Recall)")
    
    candidates = []
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        # 计算总体准确率
        acc = accuracy_score(y_true, y_pred)
        
        # 计算负类召回率 (pos_label=0)
        neg_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        
        candidates.append((thresh, acc, neg_rec))
        
        if int(thresh * 100) % 5 == 0:
            print(f"      {thresh:.3f}     | {acc:.4f} | {neg_rec:.4f}")
            
    # 1. 筛选出所有满足 Accuracy >= min_acc 的点
    valid_candidates = [x for x in candidates if x[1] >= min_acc]
    
    if valid_candidates:
        # 2. 在满足条件点中，选 Neg_Recall 最高的
        best_candidate = max(valid_candidates, key=lambda x: x[2])
        best_thresh = best_candidate[0]
        final_acc = best_candidate[1]
        final_rec = best_candidate[2]
        print(f"   [Auto-Threshold] FOUND: {best_thresh:.4f} (Acc: {final_acc:.2%}, Neg_Recall: {final_rec:.2%})")
        return best_thresh
    else:
        # 3. 如果没有任何点满足准确率要求，则退回到 Acc 最高的点（防止模型完全不可用）
        print("   [Auto-Threshold] WARNING: No threshold met min accuracy. Reverting to Max Accuracy.")
        best_candidate = max(candidates, key=lambda x: x[1])
        return best_candidate[0]

def calculate_failure_miss_rate(y_true_steps, pred_prob, threshold):
    y_true_steps = y_true_steps.flatten()
    pred_prob = pred_prob.flatten()
    actual_failures_mask = (y_true_steps > 6.0)
    total_failures = np.sum(actual_failures_mask)
    if total_failures == 0: return 0.0
    false_wins = (pred_prob[actual_failures_mask] > threshold)
    return np.sum(false_wins) / total_failures

def evaluate_model(model, Xs, fixed_threshold=None):
    X_seq, X_wid, X_bias, X_diff, X_guess_seq, y_steps, y_succ = Xs
    pred_steps, pred_prob = model.predict({
        "input_history": X_seq, "input_word_id": X_wid,
        "input_user_bias": X_bias, "input_difficulty": X_diff,
        "input_guess_sequence": X_guess_seq
    }, batch_size=1024, verbose=1)

    pred_steps = pred_steps.flatten()
    pred_prob = pred_prob.flatten()

    mae = mean_absolute_error(y_steps, np.clip(pred_steps, 0, 7))
    # [Restored] RMSE
    rmse = np.sqrt(mean_squared_error(y_steps, np.clip(pred_steps, 0, 7)))

    from predict import calculate_auc_best
    auc, used_prob, _ = calculate_auc_best(y_succ, pred_prob)

    if fixed_threshold is None:
        print("   > Finding optimal threshold (Accuracy Constrained)...")
        used_threshold = find_optimal_threshold(y_succ, pred_prob)
    else:
        print(f"   > Using provided fixed threshold: {fixed_threshold:.4f}")
        used_threshold = fixed_threshold

    # Predictions based on threshold
    y_pred_smart = (pred_prob >= used_threshold).astype(int)
    y_pred_naive = (pred_prob >= 0.5).astype(int)

    # Metrics
    naive_acc = accuracy_score(y_succ.astype(int), y_pred_naive)
    smart_acc = accuracy_score(y_succ.astype(int), y_pred_smart)
    
    naive_fmr = calculate_failure_miss_rate(y_steps, pred_prob, 0.5)
    smart_fmr = calculate_failure_miss_rate(y_steps, pred_prob, used_threshold)

    # [NEW] Negative Class Metrics (Class 0 = Failure/Loss)
    neg_precision = precision_score(y_succ.astype(int), y_pred_smart, pos_label=0, zero_division=0)
    neg_recall = recall_score(y_succ.astype(int), y_pred_smart, pos_label=0, zero_division=0)

    print(f"   [Metrics] MAE={mae:.4f}, RMSE={rmse:.4f}, AUC={auc:.4f}")
    print(f"   [Smart {used_threshold:.3f}] ACC={smart_acc:.4f}, MissRate={smart_fmr:.2%}")
    print(f"   [Negative Class] Precision={neg_precision:.4f}, Recall={neg_recall:.4f}")

    return mae, rmse, smart_acc, auc, used_threshold, smart_fmr, naive_fmr, neg_precision, neg_recall

def compute_large_error_rate(y_true, y_pred, threshold):
    errors = np.abs(y_true - y_pred)
    return np.mean(errors > threshold)

def plot_loss(history, save_path_base):
    save_dir = os.path.dirname(save_path_base) or "."
    plt.figure(figsize=(6, 6))
    plt.plot(history.history['loss'], label='Training Total Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Total Loss')
    plt.title('Training and Validation Total Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "Transformer_total_loss_curve.png"), dpi=300, bbox_inches='tight')
    plt.close()

class WandbEpochLogger(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs:
            wandb.log({k: float(v) for k, v in logs.items()}, step=epoch)

# ==========================================================
# Main
# ==========================================================
def main_train():
    set_seed(SEED)
    ensure_dirs()
    wandb.init(project="word-difficulty-prediction", name="transformer-acc-constrained", settings=wandb.Settings(_disable_stats=True))

    try:
        if hasattr(wandb.run, "summary") and "graph" in wandb.run.summary:
            wandb.run.summary.pop("graph", None)
    except Exception:
        pass

    use_cols_list = ["Game", "Trial", "Username", "target", "processed_text"]
    train_df = safe_read_csv(TRAIN_FILE, usecols=use_cols_list)
    val_df = safe_read_csv(VAL_FILE, usecols=use_cols_list)
    test_df = safe_read_csv(TEST_FILE, usecols=use_cols_list)

    user_map = {}
    if os.path.exists(PLAYER_FILE):
        pdf = pd.read_csv(PLAYER_FILE)
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
    
    diff_map = {}
    if os.path.exists(DIFFICULTY_FILE):
        df_diff = pd.read_csv(DIFFICULTY_FILE)
        diff_map = dict(zip(df_diff["word"], df_diff["avg_trial"]))

    tokenizer = fit_tokenizer(train_df)
    train_df = attach_features(train_df, tokenizer, user_map, diff_map)
    val_df = attach_features(val_df, tokenizer, user_map, diff_map)
    test_df = attach_features(test_df, tokenizer, user_map, diff_map)

    hist_train = build_history(train_df)
    hist_val = build_history(val_df)
    hist_test = build_history(test_df)

    X_train = create_samples(hist_train, LOOK_BACK)
    X_val = create_samples(hist_val, LOOK_BACK)
    X_test = create_samples(hist_test, LOOK_BACK)

    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(LOOK_BACK, vocab_size)
    model.summary()

    train_ds = tf.data.Dataset.from_tensor_slices((
        {"input_history": X_train[0], "input_word_id": X_train[1], "input_user_bias": X_train[2], "input_difficulty": X_train[3], "input_guess_sequence": X_train[4]},
        {"output_steps": X_train[5], "output_success": X_train[6]}
    )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {"input_history": X_val[0], "input_word_id": X_val[1], "input_user_bias": X_val[2], "input_difficulty": X_val[3], "input_guess_sequence": X_val[4]},
        {"output_steps": X_val[5], "output_success": X_val[6]}
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    wandb_logger = WandbEpochLogger()
    train_history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, callbacks=[early, wandb_logger])

    plot_loss(train_history, "visualization/Transformer_loss_curve.png")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    # Validation
    print("\n=== Validation Evaluation (Finding Aggressive Threshold) ===")
    val_mae, val_rmse, val_acc, val_auc, optimal_thresh, val_sfmr, val_nfmr, val_nprec, val_nrec = evaluate_model(model, X_val, fixed_threshold=None)

    # Plotting
    val_pred_steps, val_pred_prob = model.predict({
        "input_history": X_val[0], "input_word_id": X_val[1],
        "input_user_bias": X_val[2], "input_difficulty": X_val[3],
        "input_guess_sequence": X_val[4]
    }, batch_size=1024, verbose=0)
    plot_roc_curve(X_val[6], val_pred_prob.flatten(), "visualization/Transformer_validation_roc_curve.png")
    plot_scatter(X_val[5], np.clip(val_pred_steps.flatten(), 0, 7), "visualization/Transformer_validation_scatter.png", model_name="Transformer")

    # Test
    print(f"\n=== Test Evaluation (Applying Threshold {optimal_thresh:.4f}) ===")
    test_mae, test_rmse, test_acc, test_auc, _, test_sfmr, test_nfmr, test_nprec, test_nrec = evaluate_model(model, X_test, fixed_threshold=optimal_thresh)

    test_pred_steps, test_pred_prob = model.predict({
        "input_history": X_test[0], "input_word_id": X_test[1],
        "input_user_bias": X_test[2], "input_difficulty": X_test[3],
        "input_guess_sequence": X_test[4]
    }, batch_size=1024, verbose=0)
    plot_roc_curve(X_test[6], test_pred_prob.flatten(), "visualization/Transformer_test_roc_curve.png")
    plot_scatter(X_test[5], np.clip(test_pred_steps.flatten(), 0, 7), "visualization/Transformer_test_scatter.png", model_name="Transformer")

    val_large_err = compute_large_error_rate(X_val[5], np.clip(val_pred_steps.flatten(), 0, 7), LARGE_ERROR_THRESHOLD)
    test_large_err = compute_large_error_rate(X_test[5], np.clip(test_pred_steps.flatten(), 0, 7), LARGE_ERROR_THRESHOLD)

    report = f"""
========================================
 Transformer Model Report (Acc Constraint)
========================================
Constraint: Accuracy must be >= {MIN_ACCEPTABLE_ACCURACY:.1%}
Optimal Probability Threshold Found: {optimal_thresh:.4f}

---- Validation Set Metrics ----
1. Mean Absolute Error (MAE)    : {val_mae:.4f}
2. Root Mean Squared Error (RMSE)     : {val_rmse:.4f}
3. Win/Loss Prediction Accuracy        : {val_acc:.3%} (Naive 0.5: {val_nfmr:.1%})
4. Area Under ROC Curve (AUC)   : {val_auc:.4f}
5. Large Error Rate (>1.5)      : {val_large_err:.3%}
6. Negative Class Precision     : {val_nprec:.4f} (Predicted Loss correct rate)
7. Negative Class Recall        : {val_nrec:.4f} (Actual Loss detected rate)

---- Test Set Metrics ----
1. Mean Absolute Error (MAE)    : {test_mae:.4f}
2. Root Mean Squared Error (RMSE)     : {test_rmse:.4f}
3. Win/Loss Prediction Accuracy        : {test_acc:.3%}
4. Area Under ROC Curve (AUC)   : {test_auc:.4f}
5. Large Error Rate (>1.5)      : {test_large_err:.3%}
6. Negative Class Precision     : {test_nprec:.4f}
7. Negative Class Recall        : {test_nrec:.4f}
========================================
"""
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    wandb.log({
        "val_mae": val_mae, "val_rmse": val_rmse, "val_auc": val_auc, "val_smart_acc": val_acc, "val_neg_rec": val_nrec,
        "test_mae": test_mae, "test_rmse": test_rmse, "test_auc": test_auc, "test_smart_acc": test_acc, "test_neg_rec": test_nrec,
        "optimal_threshold": optimal_thresh
    })
    wandb.finish()

if __name__ == "__main__":
    main_train()