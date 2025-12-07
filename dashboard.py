"""
使用 `streamlit run dashboard.py` 命令运行仪表盘。
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
from train_LSTM import (load_tokenizer as load_lstm_tokenizer, 
                           attach_features as attach_lstm_features,
                           build_history as build_lstm_history,
                           focal_loss,
                           MODEL_SAVE_PATH as LSTM_MODEL_PATH,
                           TOKENIZER_PATH as LSTM_TOKENIZER_PATH,
                           LOOK_BACK as LSTM_LOOK_BACK,
                           MAX_TRIES as LSTM_MAX_TRIES,
                           GRID_FEAT_LEN as LSTM_GRID_SEQ_FEAT_DIM)
from train_transformer import (load_tokenizer as load_transformer_tokenizer,
                                  attach_features as attach_transformer_features,
                                  build_history as build_transformer_history,
                                  TransformerBlock,
                                  MODEL_SAVE_PATH as TRANSFORMER_MODEL_PATH,
                                  TOKENIZER_PATH as TRANSFORMER_TOKENIZER_PATH,
                                  LOOK_BACK as TRANSFORMER_LOOK_BACK,
                                  MAX_TRIES as TRANSFORMER_MAX_TRIES,
                                  GRID_FEAT_LEN as TRANSFORMER_GRID_FEAT_LEN)

# 设置页面配置
st.set_page_config(
    page_title="Wordle Prediction Results Dashboard",
    page_icon="🎯",
    layout="wide"
)

# 读取数据集和难度文件
def load_data():
    train_df = pd.read_csv("dataset/train_data.csv", usecols=["Game", "Trial", "Username", "target", "processed_text"])
    val_df = pd.read_csv("dataset/val_data.csv", usecols=["Game", "Trial", "Username", "target", "processed_text"])
    test_df = pd.read_csv("dataset/test_data.csv", usecols=["Game", "Trial", "Username", "target", "processed_text"])
    
    # 合并所有数据用于预测
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # 读取玩家数据和难度数据
    player_df = pd.read_csv("dataset/player_data.csv") if os.path.exists("dataset/player_data.csv") else pd.DataFrame()
    difficulty_df = pd.read_csv("dataset/difficulty.csv") if os.path.exists("dataset/difficulty.csv") else pd.DataFrame()
    
    return all_df, player_df, difficulty_df

# 加载 LSTM 模型
def load_lstm_model():
    if not os.path.exists(LSTM_MODEL_PATH):
        st.error("LSTM model file does not exist, please train the model first.")
        return None
    
    try:
        model = tf.keras.models.load_model(
            LSTM_MODEL_PATH,
            custom_objects={
                'focal_loss(gamma=2.0,alpha=0.25)': focal_loss(alpha=0.25, gamma=2.0)
            }
        )
        return model
    except Exception as e:
        st.error(f"Failed to load LSTM model: {e}")
        return None

# 加载 Transformer 模型
def load_transformer_model():
    if not os.path.exists(TRANSFORMER_MODEL_PATH):
        st.error("Transformer model file does not exist, please train the model first.")
        return None
    
    try:
        model = tf.keras.models.load_model(
            TRANSFORMER_MODEL_PATH,
            custom_objects={'TransformerBlock': TransformerBlock}
        )
        return model
    except Exception as e:
        st.error(f"Failed to load Transformer model: {e}")
        return None

# 随机选择玩家
def get_random_player(df):
    return random.choice(df["Username"].unique())

# LSTM 预测函数
def lstm_predict(model, tokenizer, df, user_id, user_map, diff_map):
    # 准备数据
    try:
        df = attach_lstm_features(df, tokenizer, user_map, diff_map)
    except TypeError:
        # 兼容旧版本接口，如果 train_LSTM 未更新
        df = attach_lstm_features(df, tokenizer, user_map)
        
    hist = build_lstm_history(df)
    
    if user_id not in hist:
        st.error(f"User {user_id} has no records")
        return None
    
    events = hist[user_id]
    if len(events) < 1:
        st.error("Insufficient history")
        return None
    
    # 准备输入
    if len(events) < LSTM_LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        pad_event = (avg, 0, 4.0, 4.0, np.zeros((LSTM_MAX_TRIES, LSTM_GRID_SEQ_FEAT_DIM), dtype=np.float32))
        pad = [pad_event] * (LSTM_LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-LSTM_LOOK_BACK:]
    
    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, LSTM_LOOK_BACK, 2)
    
    last = events[-1]
    wid = np.array([[last[1]]], np.int32)         
    bias = np.array([[last[2] / 7.0]], np.float32)
    diff = np.array([[last[3] / 7.0]], np.float32)
    grid_seq = last[4].reshape(1, LSTM_MAX_TRIES, LSTM_GRID_SEQ_FEAT_DIM) 
    
    # 进行预测
    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_difficulty": diff, # 添加难度输入
        "input_grid_sequence": grid_seq
    }, verbose=0)
    
    # 获取实际结果
    actual_steps = last[0]
    actual_success = 1.0 if actual_steps <= 6 else 0.0
    
    return {
        "predicted_steps": float(np.clip(p_steps, 0, 6.99)),
        "predicted_success_prob": float(p_prob),
        "actual_steps": actual_steps,
        "actual_success": actual_success,
        "steps_deviation": abs(float(np.clip(p_steps, 0, 6.99)) - actual_steps),
        "success_prediction_correct": (float(p_prob) >= 0.5) == (actual_success == 1.0)
    }

# Transformer 预测函数 (已修复：修正索引错位)
def transformer_predict(model, tokenizer, df, user_id, diff_map, user_map):
    # 准备数据
    df = attach_transformer_features(df, tokenizer, user_map, diff_map)
    hist = build_transformer_history(df)
    
    if user_id not in hist:
        st.error(f"User {user_id} has no records")
        return None
    
    events = hist[user_id]
    if len(events) < 1:
        st.error("Insufficient history")
        return None
    
    # 准备输入
    if len(events) < TRANSFORMER_LOOK_BACK:
        avg = np.mean([e[0] for e in events])
        pad_guess_seq = np.zeros((TRANSFORMER_MAX_TRIES, TRANSFORMER_GRID_FEAT_LEN), dtype=np.float32)
        # padding event 结构: (avg, word_id, bias, difficulty, grid)
        pad = [(avg, 0, 4.0, 4.0, pad_guess_seq)] * (TRANSFORMER_LOOK_BACK - len(events))
        window = pad + events
    else:
        window = events[-TRANSFORMER_LOOK_BACK:]
    
    trials = np.array([w[0] for w in window], np.float32)
    seq = np.stack([trials/7.0, np.full_like(trials, np.std(trials)/7.0)], axis=1)
    seq = seq.reshape(1, TRANSFORMER_LOOK_BACK, 2)
    
    last = events[-1]
    
    wid = np.array([[last[1]]], np.int32)            
    bias = np.array([[last[2] / 7.0]], np.float32)   
    diff = np.array([[last[3] / 7.0]], np.float32)   
    guess_seq = last[4].reshape(1, TRANSFORMER_MAX_TRIES, TRANSFORMER_GRID_FEAT_LEN) 
    
    # 进行预测
    p_steps, p_prob = model.predict({
        "input_history": seq,
        "input_difficulty": diff,
        "input_word_id": wid,
        "input_user_bias": bias,
        "input_guess_sequence": guess_seq
    }, verbose=0)
    
    # 获取实际结果
    actual_steps = last[0]
    actual_success = 1.0 if actual_steps <= 6 else 0.0
    
    return {
        "predicted_steps": float(np.clip(p_steps, 0, 6.99)),
        "predicted_success_prob": float(p_prob),
        "actual_steps": actual_steps,
        "actual_success": actual_success,
        "steps_deviation": abs(float(np.clip(p_steps, 0, 6.99)) - actual_steps),
        "success_prediction_correct": (float(p_prob) >= 0.5) == (actual_success == 1.0)
    }

# 主应用
def main():
    st.title("🎯 Wordle Prediction Results Dashboard")
    
    # 加载数据
    all_df, player_df, difficulty_df = load_data()
    
    # 加载模型
    st.sidebar.title("Model Selection")
    model_type = st.sidebar.selectbox(
        "Select Prediction Model",
        ["LSTM", "Transformer"]
    )
    
    if model_type == "LSTM":
        model = load_lstm_model()
        tokenizer = load_lstm_tokenizer() if os.path.exists(LSTM_TOKENIZER_PATH) else None
    else:
        model = load_transformer_model()
        tokenizer = load_transformer_tokenizer() if os.path.exists(TRANSFORMER_TOKENIZER_PATH) else None
    
    if not model or not tokenizer:
        st.error("Failed to load model or tokenizer, please check if files exist.")
        return
    
    # 用户选择
    st.sidebar.title("Player Selection")
    user_ids = all_df["Username"].unique()
    
    # 初始化进程状态
    if 'selected_user' not in st.session_state:
        st.session_state.selected_user = user_ids[0] if len(user_ids) > 0 else None
    
    selected_user = st.sidebar.selectbox(
        "Select Player ID",
        user_ids,
        index=user_ids.tolist().index(st.session_state.selected_user) if st.session_state.selected_user in user_ids else 0
    )
    
    if st.sidebar.button("Randomly Select Player"):
        st.session_state.selected_user = get_random_player(all_df)
        st.rerun()  # 重新运行脚本以更新选择框
    
    # 准备映射
    user_map = {}
    if os.path.exists("dataset/player_data.csv"):
        pdf = pd.read_csv("dataset/player_data.csv")
        user_map = dict(zip(pdf["Username"], pdf["avg_trial"]))
    
    diff_map = {}
    if os.path.exists("dataset/difficulty.csv"):
        ddf = pd.read_csv("dataset/difficulty.csv")
        diff_map = dict(zip(ddf["word"], ddf["avg_trial"]))
    
    # 获取玩家玩过的所有词
    user_words = all_df[all_df["Username"] == selected_user]["target"].unique()
    
    # 初始化选词状态
    if 'selected_word' not in st.session_state:
        st.session_state.selected_word = user_words[0] if len(user_words) > 0 else None
    
    # 词选择
    selected_word = st.selectbox(
        f"Select a word played by {selected_user}",
        user_words,
        index=user_words.tolist().index(st.session_state.selected_word) if st.session_state.selected_word in user_words else 0
    )
    st.session_state.selected_word = selected_word
    
    # 进行预测
    if st.button("Make Prediction"):
        with st.spinner("Making prediction..."):
            # 过滤数据，只保留到所选词之前的数据
            user_data = all_df[all_df["Username"] == selected_user].sort_values("Game")
            if selected_word in user_data["target"].values:
                # 获取所选词的游戏记录
                selected_game = user_data[user_data["target"] == selected_word].iloc[0]
                # 过滤到所选词之前的数据
                filtered_data = user_data[user_data["Game"] <= selected_game["Game"]]
                
                if model_type == "LSTM":
                    result = lstm_predict(model, tokenizer, filtered_data, selected_user, user_map, diff_map)
                else:
                    result = transformer_predict(model, tokenizer, filtered_data, selected_user, diff_map, user_map)
            else:
                # 如果词不存在，使用所有数据
                if model_type == "LSTM":
                    result = lstm_predict(model, tokenizer, all_df, selected_user, user_map, diff_map)
                else:
                    result = transformer_predict(model, tokenizer, all_df, selected_user, diff_map, user_map)
            
            if result:
                # 显示预测结果
                st.subheader("Prediction Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Predicted Steps", round(result["predicted_steps"], 2))
                with col2:
                    st.metric("Actual Steps", result["actual_steps"])
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Predicted Result", "Success" if result["predicted_success_prob"] >= 0.5 else "Failure")
                with col2:
                    st.metric("Actual Result", "Success" if result["actual_success"] == 1.0 else "Failure")
                
                # 步数偏差和成功预测是否正确
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Steps Deviation", round(result["steps_deviation"], 2))
                with col2:
                    st.metric("Success Prediction Correct", "✅ Correct" if result["success_prediction_correct"] else "❌ Incorrect")
                
                # 可视化预测与实际结果 - 独立显示
                st.subheader("Prediction vs Actual")
                col1, col2, col3 = st.columns([1, 2, 1])  # 创建三列，中间列显示图表
                with col2:
                    fig, ax = plt.subplots(figsize=(3, 2))  # 控制图表大，防止过大
                    categories = ['Predicted Steps', 'Actual Steps']
                    values = [result["predicted_steps"], result["actual_steps"]]
                    # 柱状图宽度设置
                    ax.bar(categories, values, color=['blue', 'green'], width=0.4)
                    # 设置y轴上限，确保数值有足够的显示空间
                    max_value = max(values) * 1.2
                    ax.set_ylim(0, max_value)
                    ax.set_ylabel('Steps')
                    ax.set_title('Predicted Steps vs Actual Steps')
                    # 标注数值
                    for i, v in enumerate(values):
                        ax.text(i, v + max_value * 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
                    plt.tight_layout()  # 自动调整布局
                    st.pyplot(fig)
                
                # 单词难度展示 - 显示当前预测的单词难度
                if not difficulty_df.empty:
                    st.subheader("Current Predicted Word Difficulty Information")
                    # 使用当前选择的单词
                    current_word = selected_word
                    word_difficulty = difficulty_df[difficulty_df["word"] == current_word]["avg_trial"].values[0] if current_word in difficulty_df["word"].values else "Unknown"
                    st.metric("Current Predicted Word", current_word)
                    st.metric("Average Difficulty of This Word", round(word_difficulty, 2) if word_difficulty != "Unknown" else word_difficulty)
                
                # 玩家水平展示
                if not player_df.empty:
                    st.subheader("Player Level Information")
                    if selected_user in player_df["Username"].values:
                        player_info = player_df[player_df["Username"] == selected_user].iloc[0]
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Player Average Steps", round(player_info["avg_trial"], 2))
                            st.metric("Total Games", player_info["total_games"])
                        with col2:
                            st.metric("Success Rate", f"{round(player_info['success_rate'] * 100, 2)}%")
                            st.metric("Failure Rate", f"{round(player_info['failure_rate'] * 100, 2)}%")
                        
                        # 玩家水平可视化 - 拆分为三张独立图表
                        st.subheader("Player Level Comparison")
                        
                        # 1. 平均步数对比
                        st.write("Average Steps Comparison")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig, ax = plt.subplots(figsize=(3, 2))  # 缩小图表大小
                            avg_all_players = player_df["avg_trial"].mean()
                            
                            categories = ['Current Player', 'All Players Average']
                            values = [player_info["avg_trial"], avg_all_players]
                            # 柱状图宽度设置
                            ax.bar(categories, values, color=['blue', 'green'], width=0.4)
                            # 设置y轴上限，确保数值有足够的显示空间
                            max_value = max(values) * 1.2
                            ax.set_ylim(0, max_value)
                            ax.set_ylabel('Average Steps')
                            ax.set_title('Average Steps Comparison', fontsize=8)
                            ax.set_xticklabels(categories, fontsize=6)
                            # 标注数值
                            for i, v in enumerate(values):
                                ax.text(i, v + max_value * 0.03, f'{v:.2f}', ha='center', va='bottom', fontsize=8)
                            plt.tight_layout()  # 自动调整布局
                            st.pyplot(fig)
                        
                        # 2. 成功率对比
                        st.write("Success Rate Comparison")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig, ax = plt.subplots(figsize=(3, 2))
                            avg_all_success = player_df["success_rate"].mean()
                            
                            categories = ['Current Player', 'All Players Average']
                            values = [player_info["success_rate"] * 100, avg_all_success * 100]
                            # 柱状图宽度设置
                            ax.bar(categories, values, color=['blue', 'green'], width=0.4)
                            # 设置y轴上限，预留20%的空间，确保数值有足够的显示空间
                            max_value = max(values) * 1.2
                            ax.set_ylim(0, max_value)
                            ax.set_ylabel('Success Rate (%)')
                            ax.set_title('Success Rate Comparison', fontsize=8)
                            ax.set_xticklabels(categories, fontsize=6)
                            # 标注数值
                            for i, v in enumerate(values):
                                ax.text(i, v + max_value * 0.03, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)
                            plt.tight_layout()  # 自动调整布局
                            st.pyplot(fig)
                        
                        # 3. 失败率对比
                        st.write("Failure Rate Comparison")
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            fig, ax = plt.subplots(figsize=(3, 2))
                            avg_all_failure = player_df["failure_rate"].mean()
                            
                            categories = ['Current Player', 'All Players Average']
                            values = [player_info["failure_rate"] * 100, avg_all_failure * 100]
                            # 柱状图宽度设置
                            ax.bar(categories, values, color=['blue', 'green'], width=0.4) 
                            # 设置y轴上限，确保数值有足够的显示空间
                            max_value = max(values) * 1.2
                            ax.set_ylim(0, max_value)
                            ax.set_ylabel('Failure Rate (%)')
                            ax.set_title('Failure Rate Comparison', fontsize=8)
                            ax.set_xticklabels(categories, fontsize=6)
                            # 标注数值
                            for i, v in enumerate(values):
                                ax.text(i, v + max_value * 0.03, f'{v:.2f}%', ha='center', va='bottom', fontsize=8)
                            plt.tight_layout()  # 自动调整布局
                            st.pyplot(fig)

if __name__ == "__main__":
    main()