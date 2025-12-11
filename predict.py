"""
Wordle 难度预测模型 - 生产级预测与评估脚本
功能：
1. 加载 LSTM / Transformer 模型
2. 对 [验证集] 和 [测试集] 同时进行评估
3. 使用预设固定阈值 (Fixed Thresholds)
4. 生成详细报告 (含混淆矩阵)
5. 绘图优化：
   - ROC: 棕黄色对角线、保留少量页距
   - Scatter: 带网格、抖动、[新] +/- 1.5 浅绿色误差带、保留少量页距
"""

import os
import json
import random
import ast
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import (roc_curve, roc_auc_score, mean_absolute_error, 
                             mean_squared_error, accuracy_score, recall_score, 
                             precision_score, confusion_matrix)
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

# ==========================================================
# 1. 全局配置与固定阈值
# ==========================================================
SEED = 42
LARGE_ERROR_THRESHOLD = 1.5
LOOK_BACK = 5
MAX_TRIES = 6
GRID_FEAT_LEN = 8
OOV_TOKEN = "<OOV>"

# 生产环境使用的固定阈值 (基于训练得出的最优解)
FIXED_THRESHOLDS = {
    "LSTM": 0.6900,
    "Transformer": 0.6850
}

# 文件路径配置
PATHS = {
    "lstm_model": "models/lstm/lstm_model.keras",
    "lstm_tokenizer": "models/lstm/lstm_tokenizer.json",
    "transformer_model": "models/transformer/transformer_model.keras",
    "transformer_tokenizer": "models/transformer/transformer_tokenizer.json",
    "val_data": "dataset/val_data.csv",
    "test_data": "dataset/test_data.csv",
    "player_data": "dataset/player_data.csv",
    "difficulty_data": "dataset/difficulty.csv",
    "vis_dir": "visualization",
    "out_dir": "outputs"
}

# ==========================================================
# 2. 核心组件 (必须与训练时一致)
# ==========================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

class TransformerBlock(Layer):
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
# 3. 数据预处理
# ==========================================================

def safe_read_csv(path, usecols=None):
    if not os.path.exists(path):
        print(f"[Warning] File missing: {path}")
        return pd.DataFrame()
    return pd.read_csv(path, usecols=usecols)

def load_tokenizer(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Tokenizer not found at {path}")
    with open(path, "r", encoding="utf-8") as f:
        word_index = json.load(f)
    tk = Tokenizer(oov_token=OOV_TOKEN)
    tk.word_index = word_index
    return tk

def encode_guess_sequence(grid_cell):
    default_seq = np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32)
    if pd.isna(grid_cell): return default_seq
    try:
        if isinstance(grid_cell, (list, tuple)): grid_list = list(grid_cell)
        else:
            grid_list = ast.literal_eval(grid_cell)
            if not isinstance(grid_list, (list, tuple)): grid_list = [grid_list]
            grid_list = [str(r) for r in grid_list if isinstance(r, (str, bytes))]
    except Exception: return default_seq

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
                    elif ch == "🟨": yellows_t += 1
                    elif ch == "⬜" or ch == "⬛": grays_t += 1
            cumulative_greens += greens_t
            cumulative_yellows += yellows_t
            cumulative_grays += grays_t
            cumulative_pos_green_counts += pos_green_counts_t

        feat = np.zeros(GRID_FEAT_LEN, dtype=np.float32)
        feat[0] = cumulative_greens / norm_base_cells
        feat[1] = cumulative_yellows / norm_base_cells
        feat[2] = cumulative_grays / norm_base_cells
        for i in range(5): feat[3 + i] = cumulative_pos_green_counts[i] / norm_base_rows
        feature_sequence[t] = feat
    return feature_sequence

def attach_features(df, tokenizer, user_map, diff_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)
    
    if "processed_text" in df.columns:
        df["grid_seq_processed"] = df["processed_text"].apply(encode_guess_sequence)
    else:
        df["grid_seq_processed"] = [np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32) for _ in range(len(df))]
        
    return df

def build_history(df):
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        hist[u] = [(int(r["Trial"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    float(r["word_difficulty"]), 
                    np.array(r["grid_seq_processed"], dtype=np.float32))
                   for _, r in g.iterrows()]
    return hist

def create_samples_lstm(history, look_back):
    """LSTM: 使用 target[4] (当前局特征)"""
    X_seq, X_wid, X_bias, X_diff, X_grid_seq, y_steps, y_succ = [], [], [], [], [], [], []
    for _, events in history.items():
        if len(events) <= look_back: continue
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
            X_grid_seq.append(target[4]) 
            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)
    if not X_seq: return None
    return (np.array(X_seq, np.float32), np.array(X_wid, np.int32), 
            np.array(X_bias, np.float32), np.array(X_diff, np.float32), 
            np.array(X_grid_seq, np.float32), np.array(y_steps, np.float32), 
            np.array(y_succ, np.float32))

def create_samples_transformer(history, look_back):
    """Transformer: 使用 window[-1][4] (前一局特征)"""
    X_seq, X_wid, X_bias, X_diff, X_guess_seq, y_steps, y_succ = [], [], [], [], [], [], []
    for _, events in history.items():
        if len(events) <= look_back: continue
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
    if not X_seq: return None
    return (np.array(X_seq, np.float32), np.array(X_wid, np.int32), 
            np.array(X_bias, np.float32), np.array(X_diff, np.float32), 
            np.array(X_guess_seq, np.float32), np.array(y_steps, np.float32), 
            np.array(y_succ, np.float32))

# ==========================================================
# 4. 评估逻辑与报告输出
# ==========================================================

def calculate_metrics(y_true_steps, y_true_succ, pred_steps, pred_prob, threshold):
    y_true_steps = y_true_steps.flatten()
    y_true_succ = y_true_succ.flatten()
    pred_steps = pred_steps.flatten()
    pred_prob = pred_prob.flatten()
    
    y_pred_class = (pred_prob >= threshold).astype(int)
    acc = accuracy_score(y_true_succ, y_pred_class)
    neg_prec = precision_score(y_true_succ, y_pred_class, pos_label=0, zero_division=0)
    neg_rec = recall_score(y_true_succ, y_pred_class, pos_label=0, zero_division=0)
    
    try:
        auc = roc_auc_score(y_true_succ, pred_prob)
    except:
        auc = 0.5
        
    cm = confusion_matrix(y_true_succ, y_pred_class)
    cm_vals = cm.ravel() 
    if len(cm_vals) == 4:
        tn, fp, fn, tp = cm_vals
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    mae = mean_absolute_error(y_true_steps, np.clip(pred_steps, 0, 7))
    rmse = np.sqrt(mean_squared_error(y_true_steps, np.clip(pred_steps, 0, 7)))
    large_err = np.mean(np.abs(y_true_steps - pred_steps) > LARGE_ERROR_THRESHOLD)
    
    fail_mask = (y_true_steps > 6.0)
    total_fails = np.sum(fail_mask)
    if total_fails > 0:
        missed = np.sum(pred_prob[fail_mask] > threshold)
        miss_rate = missed / total_fails
    else:
        miss_rate = 0.0
        
    return {
        "MAE": mae, "RMSE": rmse, "LargeErr": large_err,
        "ACC": acc, "AUC": auc,
        "NegPrec": neg_prec, "NegRecall": neg_rec, "MissRate": miss_rate,
        "CM": (tn, fp, fn, tp),
        "TotalCount": len(y_true_succ)
    }

def save_and_print_report(model_name, dataset_name, metrics, threshold):
    tn, fp, fn, tp = metrics["CM"]
    total = metrics["TotalCount"] if metrics["TotalCount"] > 0 else 1
    
    p_tn = tn / total
    p_fp = fp / total
    p_fn = fn / total
    p_tp = tp / total

    report_str = f"""
============================================================
           {model_name} Evaluation Report - {dataset_name}
============================================================
Fixed Threshold Used: {threshold:.4f}
Total Samples       : {total}

[1] Confusion Matrix (With Proportions)
------------------------------------------------------------
                       Predicted Loss (0)    Predicted Win (1)
------------------------------------------------------------
Actual Loss (0)      |  TN: {tn:<5} ({p_tn:>6.2%}) |  FP: {fp:<5} ({p_fp:>6.2%})
Actual Win  (1)      |  FN: {fn:<5} ({p_fn:>6.2%}) |  TP: {tp:<5} ({p_tp:>6.2%})
------------------------------------------------------------

[2] Metrics
------------------------------------------------------------
MAE                  : {metrics['MAE']:.4f}
RMSE                 : {metrics['RMSE']:.4f}
Large Error Rate     : {metrics['LargeErr']:.2%}
Accuracy             : {metrics['ACC']:.2%}
AUC                  : {metrics['AUC']:.4f}
Failure Miss Rate    : {metrics['MissRate']:.2%}
Negative Recall      : {metrics['NegRecall']:.2%}
Negative Precision   : {metrics['NegPrec']:.4f}
============================================================
"""
    print(report_str)
    filename = f"{model_name}_{dataset_name.split()[0].lower()}_outcome.txt"
    filepath = os.path.join(PATHS["out_dir"], filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_str)
    print(f"   [Report] Saved outcome to: {filepath}")

# ==========================================================
# 5. 绘图函数
# ==========================================================

def plot_roc_curve(y_true, prob, save_path, model_name, dataset_name):
    try:
        auc = roc_auc_score(y_true, prob)
    except:
        auc = 0.5
        
    fpr, tpr, _ = roc_curve(y_true, prob) 
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f'AUC = {auc:.3f}')
    
    # 棕黄色对角线
    plt.plot([0, 1], [0, 1], color='#B8860B', linestyle='--', lw=2)
    
    plt.title(f'{model_name} {dataset_name} ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # 保留 0.1 英寸的页距显得美观
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   [Plot] Saved ROC to {save_path}")

def plot_scatter(y_true, y_pred, save_path, model_name, dataset_name):
    plt.figure(figsize=(8, 6))
    
    # 1. 绘制阴影区域（y=x±1.5）
    x_range = np.linspace(0, 8, 100)
    plt.fill_between(x_range, x_range - 1.5, x_range + 1.5, 
                     color='lightgreen', alpha=0.3, label='Acceptable Range (+/- 1.5)')

    # 2. 增加抖动幅度
    jitter = np.random.normal(0, 0.25, size=len(y_true))
    
    # 3. 绘制散点
    plt.scatter(y_true + jitter, 
                y_pred + jitter, 
                alpha=0.3, s=10, c=np.abs(y_true-y_pred))
    
    # 4. 参考线
    plt.plot([0, 8], [0, 8], 'r--', label='Perfect Fit')
    
    plt.title(f'{model_name} {dataset_name} Pred vs True')
    plt.xlabel('True Steps')
    plt.ylabel('Pred Steps')
    plt.colorbar(label='Error')
    
    # 锁定坐标轴范围，使阴影和网格更好看
    plt.xlim(0, 7.5)
    plt.ylim(0, 7.5)
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper left')
    
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    # 保留 0.1 英寸的页距显得美观
    plt.savefig(save_path, dpi=150, bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"   [Plot] Saved Scatter to {save_path}")

# ==========================================================
# 6. 主流程
# ==========================================================

def main():
    set_seed(SEED)
    os.makedirs(PATHS["vis_dir"], exist_ok=True)
    os.makedirs(PATHS["out_dir"], exist_ok=True)
    
    # 1. 加载辅助数据
    u_df = safe_read_csv(PATHS["player_data"])
    d_df = safe_read_csv(PATHS["difficulty_data"])
    u_map = dict(zip(u_df["Username"], u_df["avg_trial"])) if not u_df.empty else {}
    d_map = dict(zip(d_df["word"], d_df["avg_trial"])) if not d_df.empty else {}
    
    use_cols = ["Game", "Trial", "Username", "target", "processed_text"]
    
    # 2. 加载数据
    val_raw = safe_read_csv(PATHS["val_data"], use_cols)
    test_raw = safe_read_csv(PATHS["test_data"], use_cols)
    
    if val_raw.empty or test_raw.empty:
        print("Error: Validation or Test data missing.")
        return

    # ======================================================
    # A. LSTM 预测 (Validation + Test)
    # ======================================================
    if os.path.exists(PATHS["lstm_model"]):
        print("\n" + "="*40)
        print(" PROCESS: LSTM MODEL")
        print("="*40)
        try:
            tokenizer = load_tokenizer(PATHS["lstm_tokenizer"])
            val_df = attach_features(val_raw, tokenizer, u_map, d_map)
            test_df = attach_features(test_raw, tokenizer, u_map, d_map)
            
            val_data = create_samples_lstm(build_history(val_df), LOOK_BACK)
            test_data = create_samples_lstm(build_history(test_df), LOOK_BACK)
            
            if val_data and test_data:
                custom_objs = {'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(gamma=2.0, alpha=0.25)}
                lstm = tf.keras.models.load_model(PATHS["lstm_model"], custom_objects=custom_objs)
                threshold = FIXED_THRESHOLDS["LSTM"]
                
                # --- Validation Set ---
                print("   > Evaluating Validation Set...")
                v_steps, v_prob = lstm.predict({
                    "input_history": val_data[0], "input_word_id": val_data[1],
                    "input_user_bias": val_data[2], "input_difficulty": val_data[3],
                    "input_grid_sequence": val_data[4]
                }, batch_size=1024, verbose=0)
                
                v_metrics = calculate_metrics(val_data[5], val_data[6], v_steps, v_prob, threshold)
                save_and_print_report("LSTM", "Validation Set", v_metrics, threshold)
                plot_roc_curve(val_data[6], v_prob.flatten(), os.path.join(PATHS["vis_dir"], "LSTM_val_roc_curve.png"), "LSTM", "Validation")
                plot_scatter(val_data[5], np.clip(v_steps.flatten(), 0, 7), os.path.join(PATHS["vis_dir"], "LSTM_val_scatter.png"), "LSTM", "Validation")
                
                # --- Test Set ---
                print("   > Evaluating Test Set...")
                t_steps, t_prob = lstm.predict({
                    "input_history": test_data[0], "input_word_id": test_data[1],
                    "input_user_bias": test_data[2], "input_difficulty": test_data[3],
                    "input_grid_sequence": test_data[4]
                }, batch_size=1024, verbose=0)
                
                t_metrics = calculate_metrics(test_data[5], test_data[6], t_steps, t_prob, threshold)
                save_and_print_report("LSTM", "Test Set", t_metrics, threshold)
                plot_roc_curve(test_data[6], t_prob.flatten(), os.path.join(PATHS["vis_dir"], "LSTM_test_roc_curve.png"), "LSTM", "Test")
                plot_scatter(test_data[5], np.clip(t_steps.flatten(), 0, 7), os.path.join(PATHS["vis_dir"], "LSTM_test_scatter.png"), "LSTM", "Test")
                
        except Exception as e:
            print(f"LSTM Error: {e}")
            import traceback
            traceback.print_exc()

    # ======================================================
    # B. Transformer 预测 (Validation + Test)
    # ======================================================
    if os.path.exists(PATHS["transformer_model"]):
        print("\n" + "="*40)
        print(" PROCESS: TRANSFORMER MODEL")
        print("="*40)
        try:
            tokenizer = load_tokenizer(PATHS["transformer_tokenizer"])
            val_df = attach_features(val_raw, tokenizer, u_map, d_map)
            test_df = attach_features(test_raw, tokenizer, u_map, d_map)
            
            val_data = create_samples_transformer(build_history(val_df), LOOK_BACK)
            test_data = create_samples_transformer(build_history(test_df), LOOK_BACK)
            
            if val_data and test_data:
                custom_objs = {
                    'TransformerBlock': TransformerBlock,
                    'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(gamma=2.0, alpha=0.25)
                }
                tf_model = tf.keras.models.load_model(PATHS["transformer_model"], custom_objects=custom_objs)
                threshold = FIXED_THRESHOLDS["Transformer"]
                
                # --- Validation Set ---
                print("   > Evaluating Validation Set...")
                v_steps, v_prob = tf_model.predict({
                    "input_history": val_data[0], "input_word_id": val_data[1],
                    "input_user_bias": val_data[2], "input_difficulty": val_data[3],
                    "input_guess_sequence": val_data[4]
                }, batch_size=1024, verbose=0)
                
                v_metrics = calculate_metrics(val_data[5], val_data[6], v_steps, v_prob, threshold)
                save_and_print_report("Transformer", "Validation Set", v_metrics, threshold)
                plot_roc_curve(val_data[6], v_prob.flatten(), os.path.join(PATHS["vis_dir"], "Transformer_val_roc_curve.png"), "Transformer", "Validation")
                plot_scatter(val_data[5], np.clip(v_steps.flatten(), 0, 7), os.path.join(PATHS["vis_dir"], "Transformer_val_scatter.png"), "Transformer", "Validation")
                
                # --- Test Set ---
                print("   > Evaluating Test Set...")
                t_steps, t_prob = tf_model.predict({
                    "input_history": test_data[0], "input_word_id": test_data[1],
                    "input_user_bias": test_data[2], "input_difficulty": test_data[3],
                    "input_guess_sequence": test_data[4]
                }, batch_size=1024, verbose=0)
                
                t_metrics = calculate_metrics(test_data[5], test_data[6], t_steps, t_prob, threshold)
                save_and_print_report("Transformer", "Test Set", t_metrics, threshold)
                plot_roc_curve(test_data[6], t_prob.flatten(), os.path.join(PATHS["vis_dir"], "Transformer_test_roc_curve.png"), "Transformer", "Test")
                plot_scatter(test_data[5], np.clip(t_steps.flatten(), 0, 7), os.path.join(PATHS["vis_dir"], "Transformer_test_scatter.png"), "Transformer", "Test")

        except Exception as e:
            print(f"Transformer Error: {e}")
            import traceback
            traceback.print_exc()

    print("\nAll processes completed.")

if __name__ == "__main__":
    main()