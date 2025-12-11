"""
Wordle 难度预测模型 - 结果展示仪表盘
使用这个命令启动: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import json
import ast
from tensorflow.keras.layers import Layer, MultiHeadAttention, Dense, LayerNormalization, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer

# ==========================================================
# 1. 全局配置与路径
# ==========================================================
st.set_page_config(
    page_title="Wordle Prediction Dashboard",
    page_icon="🎯",
    layout="wide"
)

# 固定阈值 (参考 predict.py 的生产环境设定)
FIXED_THRESHOLDS = {
    "LSTM": 0.6900,
    "Transformer": 0.6850
}

# 路径配置
PATHS = {
    "lstm_model": "models/lstm/lstm_model.keras",
    "lstm_tokenizer": "models/lstm/lstm_tokenizer.json",
    "transformer_model": "models/transformer/transformer_model.keras",
    "transformer_tokenizer": "models/transformer/transformer_tokenizer.json",
    "train_data": "dataset/train_data.csv",
    "val_data": "dataset/val_data.csv",
    "test_data": "dataset/test_data.csv",
    "player_data": "dataset/player_data.csv",
    "difficulty_data": "dataset/difficulty.csv"
}

# 模型参数
LOOK_BACK = 5
MAX_TRIES = 6
GRID_FEAT_LEN = 8
OOV_TOKEN = "<OOV>"

# ==========================================================
# 2. 核心组件
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

# ==========================================================
# 3. 数据预处理工具
# ==========================================================

def load_tokenizer(path):
    if not os.path.exists(path):
        return None
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
    # 处理 Tokenizer
    if tokenizer:
        seqs = tokenizer.texts_to_sequences(df["target"])
        df["word_id"] = [s[0] if s else 0 for s in seqs]
    else:
        df["word_id"] = 0
        
    df["word_difficulty"] = df["target"].map(diff_map).fillna(4.0).astype(float)
    df["user_bias"] = df["Username"].map(user_map).fillna(4.0).astype(float)
    
    if "processed_text" in df.columns:
        df["grid_seq_processed"] = df["processed_text"].apply(encode_guess_sequence)
    else:
        df["grid_seq_processed"] = [np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32) for _ in range(len(df))]
        
    return df

def build_history(df):
    hist = {}
    if df.empty: return hist
    df_sorted = df.sort_values(["Username", "Game"])
    for u, g in df_sorted.groupby("Username", sort=False):
        hist[u] = [(int(r["Trial"]),
                    int(r["word_id"]),
                    float(r["user_bias"]),
                    float(r["word_difficulty"]), 
                    np.array(r["grid_seq_processed"], dtype=np.float32))
                   for _, r in g.iterrows()]
    return hist

# ==========================================================
# 4. 预测与加载逻辑
# ==========================================================

@st.cache_resource
def load_data_and_maps():
    # 加载 CSV
    dfs = []
    for p in [PATHS["train_data"], PATHS["val_data"], PATHS["test_data"]]:
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, usecols=["Game", "Trial", "Username", "target", "processed_text"]))
    
    all_df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    player_df = pd.read_csv(PATHS["player_data"]) if os.path.exists(PATHS["player_data"]) else pd.DataFrame()
    diff_df = pd.read_csv(PATHS["difficulty_data"]) if os.path.exists(PATHS["difficulty_data"]) else pd.DataFrame()
    
    # 构建映射
    u_map = dict(zip(player_df["Username"], player_df["avg_trial"])) if not player_df.empty else {}
    d_map = dict(zip(diff_df["word"], diff_df["avg_trial"])) if not diff_df.empty else {}
    
    return all_df, player_df, diff_df, u_map, d_map

def load_model_safe(model_type):
    # 统一的自定义对象字典
    custom_objs = {
        'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(gamma=2.0, alpha=0.25),
        'TransformerBlock': TransformerBlock, # 无论如何都包含，以避免多次定义
    }
    
    if model_type == "LSTM":
        path = PATHS["lstm_model"]
        tokenizer_path = PATHS["lstm_tokenizer"]
    else: # Transformer
        path = PATHS["transformer_model"]
        tokenizer_path = PATHS["transformer_tokenizer"]
        
    if not os.path.exists(path): return None, None
    
    tokenizer = load_tokenizer(tokenizer_path)
    model = None
    
    try:
        with tf.keras.utils.custom_object_scope(custom_objs):
            model = tf.keras.models.load_model(path, custom_objects=custom_objs)
            
    except Exception as e:
        st.error(f"Error loading {model_type} model from {path}: {e}")
        st.caption("Please ensure model files exist and are compatible with your TensorFlow/Keras version.")
        return None, None
        
    return model, tokenizer

def make_prediction(model_type, model, tokenizer, df_history, user_id, current_word_row, user_map, diff_map):
    # 1. 准备历史特征
    # 我们需要 attach features 到整个历史记录，才能构建 sequence
    full_df = pd.concat([df_history, pd.DataFrame([current_word_row])], ignore_index=True)
    full_df = attach_features(full_df, tokenizer, user_map, diff_map)
    hist_dict = build_history(full_df)
    
    if user_id not in hist_dict: return None
    events = hist_dict[user_id]
    
    # 目标是预测 events[-1] (即当前选中的词)，使用 events[:-1] 作为历史
    target_event = events[-1]
    history_events = events[:-1]
    
    # 2. 构建 Input Window
    if len(history_events) < LOOK_BACK:
        # Padding
        if len(history_events) > 0:
            avg_trial = np.mean([e[0] for e in history_events])
            std_trial = np.std([e[0] for e in history_events])
        else:
            avg_trial = 4.0
            std_trial = 0.0
            
        pad_len = LOOK_BACK - len(history_events)
        # Padding 结构
        pad_event = (avg_trial, 0, 4.0, 4.0, np.zeros((MAX_TRIES, GRID_FEAT_LEN), dtype=np.float32))
        window = [pad_event] * pad_len + history_events
    else:
        window = history_events[-LOOK_BACK:]
        
    # 3. 提取序列特征
    trials = np.array([w[0] for w in window], dtype=np.float32)
    norm = trials / 7.0
    std = np.std(trials) / 7.0
    # 形状重塑
    input_seq = np.stack([norm, np.full_like(norm, std)], axis=1).reshape(1, LOOK_BACK, 2)
    
    # 4. 提取静态特征
    wid = np.array([[target_event[1]]], dtype=np.int32)
    bias = np.array([[target_event[2] / 7.0]], dtype=np.float32)
    diff = np.array([[target_event[3] / 7.0]], dtype=np.float32)
    
    # 5. 提取 Grid/Guess Sequence
    if model_type == "LSTM":
        # LSTM 使用当前局的 grid (target_event[4])
        grid_input = target_event[4].reshape(1, MAX_TRIES, GRID_FEAT_LEN)
        inputs = {
            "input_history": input_seq, "input_word_id": wid,
            "input_user_bias": bias, "input_difficulty": diff,
            "input_grid_sequence": grid_input
        }
    else:
        # Transformer 使用前一局的 grid (window[-1][4])
        guess_input = window[-1][4].reshape(1, MAX_TRIES, GRID_FEAT_LEN)
        inputs = {
            "input_history": input_seq, "input_word_id": wid,
            "input_user_bias": bias, "input_difficulty": diff,
            "input_guess_sequence": guess_input
        }
        
    # 6. Predict
    p_steps, p_prob = model.predict(inputs, verbose=0)
    
    return {
        "pred_steps": float(np.clip(p_steps[0][0], 0, 6.99)),
        "pred_prob": float(p_prob[0][0]),
        "actual_steps": target_event[0],
        "actual_success": 1.0 if target_event[0] <= 6 else 0.0
    }

# ==========================================================
# 5. 主界面逻辑
# ==========================================================

def add_value_labels(ax):
    """辅助函数：为柱状图添加数值标签"""
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f'{height:.2f}',
                    (p.get_x() + p.get_width() / 2., height),
                    ha='center', va='bottom', 
                    xytext=(0, 2), 
                    textcoords='offset points',
                    fontsize=9, weight='bold')

def main():
    st.title("🎯 Wordle Prediction Dashboard")
    st.markdown("---")

    # 加载数据
    with st.spinner("Loading Data..."):
        all_df, player_df, diff_df, u_map, d_map = load_data_and_maps()
        
    if all_df.empty:
        st.error("Dataset files missing. Please check 'dataset/' folder.")
        return

    # 滑动选择器
    st.sidebar.header("Configuration")
    model_type = st.sidebar.radio("Select Model", ["LSTM", "Transformer"])
    
    model, tokenizer = load_model_safe(model_type)
    if not model:
        st.error(f"Model file for {model_type} not found at {PATHS['lstm_model' if model_type=='LSTM' else 'transformer_model']}.")
        return

    # 玩家选择器
    user_list = sorted(all_df["Username"].unique())
    
    # 初始化玩家选择器
    if 'selected_user_state' not in st.session_state:
        st.session_state.selected_user_state = user_list[0]

    def set_random_user():
        st.session_state.selected_user_state = random.choice(user_list)
        
    st.sidebar.button("🎲 Random Player", on_click=set_random_user)
    
    # 玩家选择框
    selected_user = st.sidebar.selectbox(
        "Select Player", 
        user_list, 
        key='selected_user_state'
    )
    
    # 筛选用户数据
    user_data = all_df[all_df["Username"] == selected_user].sort_values("Game")
    
    # 玩家选择框
    user_words = user_data["target"].unique()
    if len(user_words) == 0:
        st.warning("No games found for this user.")
        return
        
    selected_word = st.selectbox("Select Target Word to Predict", user_words)
    
    # 获取目标行数据
    target_row_idx = user_data[user_data["target"] == selected_word].index[0]
    target_row = user_data.loc[target_row_idx]
    
    # 筛选历史数据
    history_df = user_data[user_data["Game"] < target_row["Game"]]
    
    st.markdown(f"### Prediction for User: **{selected_user}** | Word: **{selected_word}**")
    
    if st.button("🚀 Run Prediction"):
        with st.spinner("Calculating..."):
            result = make_prediction(
                model_type, model, tokenizer, 
                history_df, selected_user, target_row, 
                u_map, d_map
            )
            
        if result:
            threshold = FIXED_THRESHOLDS[model_type]
            is_win_pred = result["pred_prob"] >= threshold
            is_win_actual = result["actual_success"] == 1.0
            
            # 指标展示
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Predicted Steps", f"{result['pred_steps']:.2f}")
                st.metric("Actual Steps", f"{result['actual_steps']}")
                
            with col2:
                # prob_color = "green" if is_win_pred else "red"
                st.metric("Win Probability", f"{result['pred_prob']:.2%}")
                st.caption(f"Threshold: {threshold:.4f}")
                
            with col3:
                res_str = "WIN" if is_win_pred else "LOSS"
                actual_str = "WIN" if is_win_actual else "LOSS"
                st.metric("Prediction Outcome", res_str, delta="Correct" if res_str==actual_str else "Wrong")
                st.metric("Actual Outcome", actual_str)

            # 可视化分析
            st.markdown("---")
            st.subheader("Visual Analysis")
            
            c1, c2, c3 = st.columns(3)
            
            # 1. 步数对比
            with c1:
                fig, ax = plt.subplots(figsize=(4, 3))
                bar_plot = ax.bar(["Pred", "Actual"], [result["pred_steps"], result["actual_steps"]], 
                       color=['#3498db', '#95a5a6'])
                ax.set_title("Steps Comparison")
                ax.set_ylim(0, 8)
                add_value_labels(ax) # 添加标签
                st.pyplot(fig)
                
            # 2. 成功概率与阈值对比
            with c2:
                fig, ax = plt.subplots(figsize=(4, 3))
                bar_plot = ax.bar(["Win Prob"], [result["pred_prob"]], color=['#2ecc71' if is_win_pred else '#e74c3c'])
                ax.axhline(y=threshold, color='black', linestyle='--', label=f'Thresh {threshold}')
                ax.set_ylim(0, 1.1) # 留出标签空间
                ax.set_title("Success Probability")
                ax.legend(loc='lower center')
                add_value_labels(ax) # 添加标签
                st.pyplot(fig)

            # 3. 实际步数 vs 单词难度
            with c3:
                # 获取该单词的平均步数 (Difficulty)
                word_avg_steps = d_map.get(selected_word, 4.0)
                # 获取该玩家本次的实际步数
                actual_steps = result['actual_steps']
                
                fig, ax = plt.subplots(figsize=(4, 3))
                # 修改对比对象：实际步数 vs 单词平均难度
                bar_plot = ax.bar(["Actual Steps", "Word Diff"], [actual_steps, word_avg_steps], 
                                  color=['#95a5a6', '#e67e22'])
                ax.set_title("Performance vs Difficulty")
                ax.set_ylabel("Steps")
                ax.set_ylim(0, 8)
                add_value_labels(ax) # 添加标签
                st.pyplot(fig)

if __name__ == "__main__":
    main()