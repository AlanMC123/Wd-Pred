#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重构版 LSTM 多输入预测脚本（稳定版）
不需要 CLI 或 parse_args，直接运行即开始训练。
预测模式可以通过修改 main() 下方的一行开关来启用。
"""

import os
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import wandb
from wandb.integration.keras import WandbCallback
from typing import Dict, Tuple, List
from tensorflow.keras.layers import (Input, Dense, Dropout, Embedding, Flatten,
                                     LSTM, Bidirectional, Concatenate)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
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

LOOK_BACK = 8
BATCH_SIZE = 1024
EPOCHS = 15
LEARNING_RATE = 0.001

MODEL_SAVE_PATH = "models/lstm/lstm_model.keras"
TOKENIZER_PATH = "models/lstm/lstm_tokenizer.json"

# LSTM 架构参数
LSTM_UNITS = 64
DROPOUT_RATE = 0.15
EMBEDDING_DIM = 24

OOV_TOKEN = "<OOV>"

LARGE_ERROR_THRESHOLD = 1.5
PATIENCE = 5
REPORT_SAVE_PATH = "outputs/lstm_output.txt"

# 固定随机种子
SEED = 42

# ==========================================================
# 工具函数
# ==========================================================

def set_seed(seed):
    """设置所有随机种子以确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # 设置确定性操作
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


def attach_features(df, tokenizer, diff_map, user_map):
    df = df.copy()
    df["target"] = df["target"].astype(str)
    seqs = tokenizer.texts_to_sequences(df["target"])
    df["word_id"] = [s[0] if s else 0 for s in seqs]
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)
    return df


def build_history(df) -> Dict[str, List[Tuple]]:
    hist = {}
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        hist[u] = [(int(r["Trial"]),
                    float(r["word_difficulty"]),
                    int(r["word_id"]),
                    float(r["user_bias"]))
                   for _, r in g.iterrows()]
    return hist


def create_samples(history, look_back):
    X_seq, X_diff, X_wid, X_bias, y_steps, y_succ = [], [], [], [], [], []
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
            X_diff.append([target[1] / 7.0])
            X_wid.append([target[2]])
            X_bias.append([target[3] / 7.0])
            y_steps.append(min(float(target[0]), 7.0))
            y_succ.append(1.0 if target[0] <= 6 else 0.0)

    if not X_seq:
        return (np.zeros((0, look_back, 2), np.float32),
                np.zeros((0, 1), np.float32),
                np.zeros((0, 1), np.int32),
                np.zeros((0, 1), np.float32),
                np.zeros((0,), np.float32),
                np.zeros((0,), np.float32))

    return (
        np.array(X_seq, np.float32),
        np.array(X_diff, np.float32),
        np.array(X_wid, np.int32),
        np.array(X_bias, np.float32),
        np.array(y_steps, np.float32),
        np.array(y_succ, np.float32)
    )


# ==========================================================
# LSTM 模型
# ==========================================================

def build_model(look_back, vocab_size):
    # 历史输入分支
    h_in = Input((look_back, 2), name="input_history")
    # 使用双向LSTM处理历史序列
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

    # 合并所有特征
    z = Concatenate()([x, d1, wemb, b1])
    z = Dense(64, activation="relu")(z)
    z = Dropout(DROPOUT_RATE)(z)

    # 输出层
    out_steps = Dense(1, "linear", name="output_steps")(Dense(32, "relu")(z))
    out_succ = Dense(1, "sigmoid", name="output_success")(Dense(16, "relu")(z))

    model = Model([h_in, diff_in, wid_in, bias_in], [out_steps, out_succ])
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
    X_seq, X_diff, X_wid, X_bias, y_steps, y_succ = Xs
    pred_steps, pred_prob = model.predict({
        "input_history": X_seq,
        "input_difficulty": X_diff,
        "input_word_id": X_wid,
        "input_user_bias": X_bias
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
    """Plot ROC AUC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc_score(y_true, y_pred):.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
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
    """Plot training and validation loss curves"""
    plt.figure(figsize=(12, 6))
    
    # Plot total loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
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
        plt.plot(history.history['val_output_steps_loss'], label='Validation Steps Loss')
    if 'output_success_loss' in history.history:
        plt.plot(history.history['output_success_loss'], label='Training Success Loss')
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
# 主程序（无需 parse_args）
# ==========================================================

def main_train():
    # 设置随机种子确保可重复性
    set_seed(SEED)
    
    # 初始化wandb
    wandb.init(
        project="word-difficulty-prediction",
        name="lstm-model-run",
        config={
            "model_type": "LSTM",
            "look_back": LOOK_BACK,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lstm_units": LSTM_UNITS,
            "dropout_rate": DROPOUT_RATE,
            "embedding_dim": EMBEDDING_DIM,
            "seed": SEED
        }
    )
    
    ensure_dirs()

    # 1. 数据读取
    train_df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target"])
    val_df = safe_read_csv(VAL_FILE, usecols=["Game", "Trial", "Username", "target"])
    test_df = safe_read_csv(TEST_FILE, usecols=["Game", "Trial", "Username", "target"])

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

    train_df = attach_features(train_df, tokenizer, diff_map, user_map)
    val_df = attach_features(val_df, tokenizer, diff_map, user_map)
    test_df = attach_features(test_df, tokenizer, diff_map, user_map)

    # 4. Build histories
    hist_train = build_history(train_df)
    hist_val = build_history(val_df)
    hist_test = build_history(test_df)

    # 5. Sliding samples
    X_train = create_samples(hist_train, LOOK_BACK)
    X_val = create_samples(hist_val, LOOK_BACK)
    X_test = create_samples(hist_test, LOOK_BACK)

    print(f"Train={len(X_train[0])}, Val={len(X_val[0])}, Test={len(X_test[0])}")

    vocab_size = len(tokenizer.word_index) + 1

    # 6. Model
    model = build_model(LOOK_BACK, vocab_size)
    model.summary()

    # 7. TF dataset
    train_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_train[0],
            "input_difficulty": X_train[1],
            "input_word_id": X_train[2],
            "input_user_bias": X_train[3]
        },
        {
            "output_steps": X_train[4],
            "output_success": X_train[5]
        }
    )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((
        {
            "input_history": X_val[0],
            "input_difficulty": X_val[1],
            "input_word_id": X_val[2],
            "input_user_bias": X_val[3]
        },
        {
            "output_steps": X_val[4],
            "output_success": X_val[5]
        }
    )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # 8. 训练
    early = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)
    train_history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=EPOCHS, 
        callbacks=[early, WandbCallback(save_model=False, log_model=False)]
    )
    
    # 绘制并保存损失曲线
    loss_curve_path = "visualization/LSTM_loss_curve.png"
    plot_loss(train_history, loss_curve_path)

    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to {MODEL_SAVE_PATH}")

    print("\n=== Validation ===")
    val_mae, val_rmse, val_acc, val_auc = evaluate_model(model, X_val)
    
    # 记录验证集指标到wandb
    wandb.log({
        "val_mae": val_mae,
        "val_rmse": val_rmse,
        "val_accuracy": val_acc,
        "val_auc": val_auc
    })
    
    # 绘制验证集AUC曲线
    val_pred_steps, val_pred_prob = model.predict({
        "input_history": X_val[0],
        "input_difficulty": X_val[1],
        "input_word_id": X_val[2],
        "input_user_bias": X_val[3]
    }, batch_size=1024, verbose=0)
    val_roc_curve_path = "visualization/LSTM_validation_roc_curve.png"
    plot_roc_curve(X_val[5], val_pred_prob.flatten(), val_roc_curve_path)

    print("\n=== Test ===")
    test_mae, test_rmse, test_acc, test_auc = evaluate_model(model, X_test)
    
    # 记录测试集指标到wandb
    wandb.log({
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_accuracy": test_acc,
        "test_auc": test_auc
    })
    
    # 绘制测试集AUC曲线
    test_pred_steps, test_pred_prob = model.predict({
        "input_history": X_test[0],
        "input_difficulty": X_test[1],
        "input_word_id": X_test[2],
        "input_user_bias": X_test[3]
    }, batch_size=1024, verbose=0)
    test_roc_curve_path = "visualization/LSTM_test_roc_curve.png"
    plot_roc_curve(X_test[5], test_pred_prob.flatten(), test_roc_curve_path)

    # --------------------------------------------------------
    # 生成大型误差统计
    # --------------------------------------------------------
    val_pred_steps, _ = model.predict({
        "input_history": X_val[0],
        "input_difficulty": X_val[1],
        "input_word_id": X_val[2],
        "input_user_bias": X_val[3]
    }, batch_size=1024, verbose=0)
    val_pred_steps = val_pred_steps.flatten()
    val_large_error_rate = compute_large_error_rate(X_val[4], np.clip(val_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

    test_pred_steps, _ = model.predict({
        "input_history": X_test[0],
        "input_difficulty": X_test[1],
        "input_word_id": X_test[2],
        "input_user_bias": X_test[3]
    }, batch_size=1024, verbose=0)
    test_pred_steps = test_pred_steps.flatten()
    test_large_error_rate = compute_large_error_rate(X_test[4], np.clip(test_pred_steps, 0, 7), LARGE_ERROR_THRESHOLD)

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

    # --------------------------------------------------------
    # 保存到文件
    # --------------------------------------------------------
    with open(REPORT_SAVE_PATH, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n📄 Report saved to: {REPORT_SAVE_PATH}")
    print(report)
    
    # 记录大型误差率到wandb
    wandb.log({
        "val_large_error_rate": val_large_error_rate,
        "test_large_error_rate": test_large_error_rate
    })
    
    # 结束wandb运行
    wandb.finish()


# 预测模式（按需启用）
def main_predict(user_id):
    if not os.path.exists(MODEL_SAVE_PATH):
        raise FileNotFoundError("请先训练模型。")

    model = tf.keras.models.load_model(MODEL_SAVE_PATH)

    tokenizer = load_tokenizer()

    df = safe_read_csv(TRAIN_FILE, usecols=["Game", "Trial", "Username", "target"])
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
        pad = [(avg, 4.0, 0, 4.0)] * (LOOK_BACK - len(events))
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

    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_difficulty": diff,
        "input_word_id": wid,
        "input_user_bias": bias
    }, verbose=0)

    print(f"预测步数: {float(np.clip(p_steps, 0, 6.99)):.2f}")
    print(f"成功概率: {float(p_prob):.3f}")


# ==========================================================
# 程序启动入口（只需要改这几行即可控制 train 或 predict）
# ==========================================================
if __name__ == "__main__":
    # 检测模型是否存在，如果存在则直接进行预测
    if os.path.exists(MODEL_SAVE_PATH) and os.path.exists(TOKENIZER_PATH):
        print("检测到已存在模型，直接进行预测...")
        USER_TO_PREDICT = "Alice"  # 若要预测，填用户 ID
        main_predict(USER_TO_PREDICT)
    else:
        # 运行模式
        RUN_MODE = "train"     # "train" 或 "predict"
        USER_TO_PREDICT = "Alice"  # 若要预测，填用户 ID
        
        if RUN_MODE == "train":
            print("未检测到模型或tokenizer，开始训练模型...")
            main_train()
        else:
            main_predict(USER_TO_PREDICT)