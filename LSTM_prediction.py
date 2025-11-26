import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import set_global_policy
import random
import os
import io

# ==========================================
# 0. 用户配置参数 (STRICT CONFIG)
# ==========================================
FILE_PATH = 'F:\Codes\wordle_games.csv'
MAX_ROWS = 6923127
LOOK_BACK = 5       
EPOCHS = 10         
BATCH_SIZE = 1024    
PREDICTION_SAMPLE_SIZE = 10 
VALIDATION_SAMPLE_SIZE = 100000 
LARGE_ERROR_THRESHOLD = 1.5 
EMBEDDING_DIM = 32
MODEL_SAVE_PATH = 'LSTM_Model'

# ==========================================
# 1. 数据加载与高级特征工程
# ==========================================

def process_data_and_extract_features(file_path, nrows):
    """读取数据，计算单词难度、单词ID、用户平均偏好。"""
    print("Step 1: 读取数据并构建四特征...")
    try:
        df = pd.read_csv(file_path, nrows=nrows, usecols=['Game', 'Trial', 'Username', 'target'])
        df = df.dropna()
    except:
        # 模拟数据... (省略模拟逻辑，见前一个版本)
        print("⚠️ 文件未找到，请确保 wordle_games.csv 存在。")
        return {}, 2, None, None 

    # --- Feature A & B: 单词难度 (Difficulty) & 单词 ID (Embedding) ---
    word_stats = df.groupby('target')['Trial'].mean().to_dict()
    df['word_difficulty'] = df['target'].map(word_stats)
    tokenizer = Tokenizer(); tokenizer.fit_on_texts(df['target'])
    df['word_id'] = df['target'].apply(lambda x: tokenizer.texts_to_sequences([x])[0][0])
    vocab_size = len(tokenizer.word_index) + 1

    # --- Feature C: 用户偏好 (User Bias) ---
    # 计算每个用户的全局平均步数
    user_stats = df.groupby('Username')['Trial'].mean().to_dict()
    df['user_bias'] = df['Username'].map(user_stats)

    df = df.sort_values(by=['Username', 'Game'])
    
    # 构建复合字典：(Trial, Difficulty, WordID, UserBias)
    history_map = df.groupby('Username').apply(
        lambda x: list(zip(x['Trial'], x['word_difficulty'], x['word_id'], x['user_bias']))
    ).to_dict()
    
    return history_map, vocab_size, tokenizer, word_stats

def create_multi_input_dataset(user_history_map, look_back):
    """构建四输入数据集：X_seq, X_diff, X_word, X_bias."""
    print("Step 2: 构建四输入样本...")
    
    X_seq_list = []
    X_diff_list = []; X_word_list = []; X_bias_list = []
    y_steps = []; y_success = []
    valid_players = []

    for user, history in user_history_map.items():
        if len(history) > look_back:
            for i in range(len(history) - look_back):
                
                # --- 1. 历史序列特征 (LSTM Input) ---
                past_trials = [h[0] for h in history[i : i+look_back]]
                past_arr = np.array(past_trials)
                std_dev = np.std(past_arr) / 7.0; seq_norm = past_arr / 7.0
                seq_2d = np.stack([seq_norm, np.full_like(seq_norm, std_dev)], axis=1)
                
                # --- 2. 目标信息和用户偏好 (Context Inputs) ---
                target_game = history[i + look_back]
                target_trial = target_game[0]
                target_difficulty = target_game[1]
                target_word_id = target_game[2]
                target_user_bias = target_game[3] # 新增
                
                # 收集数据
                X_seq_list.append(seq_2d)
                X_diff_list.append(target_difficulty / 7.0) 
                X_word_list.append(target_word_id)
                X_bias_list.append(target_user_bias / 7.0) # 新增
                
                # 标签
                y_steps.append(7.0 if target_trial > 6 else float(target_trial))
                y_success.append(1.0 if target_trial <= 6 else 0.0)
            
            valid_players.append(user)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_diff_list, dtype=np.float32),
        np.array(X_word_list, dtype=np.float32),
        np.array(X_bias_list, dtype=np.float32), # 新增输出
        np.array(y_steps, dtype=np.float32),
        np.array(y_success, dtype=np.float32),
        valid_players
    )

# ==========================================
# 2. 模型构建 (Multi-Input Four Branch)
# ==========================================

def build_context_model(look_back, vocab_size, embedding_dim):
    print(f"Step 3: 构建四输入神经网络 (Vocab={vocab_size})...")
    
    # --- Input 1: 玩家历史 (LSTM) ---
    input_hist = Input(shape=(look_back, 2), name='input_history')
    x1 = LSTM(128, return_sequences=False)(input_hist); x1 = Dropout(0.3)(x1)
    
    # --- Input 2: 单词难度 (Dense) ---
    input_diff = Input(shape=(1,), name='input_difficulty'); x2 = Dense(16, activation='relu')(input_diff)
    
    # --- Input 3: 单词 ID (Embedding) ---
    input_word = Input(shape=(1,), name='input_word_id'); x3 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)(input_word); x3 = Flatten()(x3)
    
    # --- Input 4: 用户偏好 (Dense) ---
    input_bias = Input(shape=(1,), name='input_user_bias'); x4 = Dense(16, activation='relu')(input_bias) # 新增
    
    # --- 融合层 (Concatenate) ---
    combined = Concatenate()([x1, x2, x3, x4]) # 融合四个分支
    
    z = Dense(64, activation='relu')(combined); z = Dropout(0.2)(z)
    
    # --- 输出层 ---
    out_steps = Dense(1, name='output_steps', dtype='float32')(Dense(32, activation='relu')(z))
    out_success = Dense(1, activation='sigmoid', name='output_success', dtype='float32')(Dense(16, activation='relu')(z))
    
    # 必须更新 inputs 列表
    model = Model(inputs=[input_hist, input_diff, input_word, input_bias], outputs=[out_steps, out_success])
    
    model.compile(optimizer='adam',
                  loss={'output_steps': 'mse', 'output_success': 'binary_crossentropy'},
                  loss_weights={'output_steps': 1.0, 'output_success': 0.5},
                  metrics={'output_success': 'accuracy'})
    return model

# ==========================================
# 3. 验证与回测逻辑 (修正后的版本)
# ==========================================

def prepare_single_inference_input(history_tuples, target_diff, target_word_id, target_user_bias, look_back):
    """辅助函数：为单次预测准备四输入张量。"""
    trials = [h[0] for h in history_tuples]
    arr = np.array(trials)
    
    # 构造 Input 1 (History)
    std = np.std(arr) / 7.0; norm = arr / 7.0
    seq_2d = np.stack([norm, np.full_like(norm, std)], axis=1).reshape(1, look_back, 2)
    # 构造 Input 2, 3, 4 (Context)
    diff_in = np.array([target_diff / 7.0]).reshape(1, 1)
    word_in = np.array([target_word_id]).reshape(1, 1)
    bias_in = np.array([target_user_bias / 7.0]).reshape(1, 1)
    
    return [
        seq_2d.astype(np.float32), 
        diff_in.astype(np.float32), 
        word_in.astype(np.float32), 
        bias_in.astype(np.float32)
    ]

def evaluate_model_and_get_preds(model, val_inputs, val_labels):
    """进行批量预测，显示进度条，并计算整体指标。"""
    print("\nStep 4: 开始批量预测，显示进度条...")
    
    # 批量预测 (verbose=1 开启进度条)
    predictions = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=1)
    
    pred_steps = predictions[0].flatten()
    pred_success_prob = predictions[1].flatten()
    
    # 真实标签
    true_steps = val_labels['output_steps']
    
    # 计算指标
    mae = mean_absolute_error(true_steps, pred_steps.clip(max=7.0))
    
    return pred_steps.clip(max=6.99), pred_success_prob, mae

def perform_validation(model, user_history_map, valid_players, look_back, sample_size, threshold, pred_steps_full):
    
    buffer = io.StringIO()
    
    eligible = [u for u in valid_players if len(user_history_map[u]) >= look_back + 1]
    if len(eligible) < sample_size: sample_size = len(eligible)
    
    buffer.write(f"\nStep 5: 启动回测验证抽样报告 (样本数={sample_size}, 输出至 output.txt)...")
    
    header = f"{'User ID':<10} | {'Bias':<5} | {'Diff':<5} | {'Real':<5} | {'Pred':<5} | {'Err':<5} | {'Status'}"
    buffer.write("\n" + "-" * 60 + "\n" + header + "\n" + "-" * 60)
    
    # 随机抽取用户来展示他们的最后一次游戏预测
    report_users = random.sample(eligible, min(10, sample_size))
    
    large_errors = 0
    
    for i, user in enumerate(report_users):
        full_hist = user_history_map[user]
        # 获取目标数据 (Trial, Difficulty, WordID, UserBias)
        target_data = full_hist[-1]
        
        real_trial = float(target_data[0])
        t_diff = target_data[1]; t_word = target_data[2]; t_bias = target_data[3]
        
        # 临时进行单样本预测以获得该用户的预测值 (此步骤效率低，仅为生成报告示例)
        temp_inputs = prepare_single_inference_input(full_hist[-(look_back+1) : -1], t_diff, t_word, t_bias, look_back)
        p_steps, _ = model.predict(temp_inputs, verbose=0)
        
        pred_val = min(p_steps[0][0], 6.99)
        err = abs(pred_val - real_trial)
        
        if err > threshold: large_errors += 1
        
        status = "✅" if err < 1.0 else "⚠️"
        if err > threshold: status = "❌"
        line = f"{str(user):<10} | {t_bias:.2f}  | {t_diff:.2f}  | {real_trial:.0f}    | {pred_val:.2f}  | {err:.2f}  | {status}"
        buffer.write(f"\n{line}")

    if sample_size > len(report_users):
        buffer.write(f"\n... (其余 {sample_size - len(report_users)} 条省略)")
    
    # 打印到控制台
    print(buffer.getvalue())
    
    # 写入文件
    with open("output.txt", "a", encoding="utf-8") as f: # 改为 'a' 以追加到全局报告
        f.write(buffer.getvalue())
    print("\n✅ 抽样报告已成功导出至 output.txt 文件。")


# ==========================================
# 4. 主程序 (Main)
# ==========================================

def main():
    # --- GPU 设置 ---
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
            set_global_policy('mixed_float16')
            print("✅ GPU 加速已开启 (Mixed Float16)")
        except: pass
        
    # 1. 数据处理
    user_map, vocab_size, _, _ = process_data_and_extract_features(FILE_PATH, MAX_ROWS)
    if not user_map: return 

    # 2. 数据集构建
    X_s, X_d, X_w, X_b, y_st, y_su, valid_users = create_multi_input_dataset(user_map, LOOK_BACK)
    
    # 划分训练/验证集
    indices = np.arange(len(y_st))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42)  # 训练集:测试集 = 8:2
    
    # 构建 TF Dataset 辅助函数
    def make_ds(idx):
        return tf.data.Dataset.from_tensor_slices((
            {
                'input_history': X_s[idx],
                'input_difficulty': X_d[idx],
                'input_word_id': X_w[idx],
                'input_user_bias': X_b[idx]
            },
            {
                'output_steps': y_st[idx],
                'output_success': y_su[idx]
            }
        )).shuffle(20000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
        
    train_ds = make_ds(train_idx)
    # val_ds 用于模型评估，不需要 shuffle
    val_ds = tf.data.Dataset.from_tensor_slices((
            {
                'input_history': X_s[val_idx],
                'input_difficulty': X_d[val_idx],
                'input_word_id': X_w[val_idx],
                'input_user_bias': X_b[val_idx]
            },
            {
                'output_steps': y_st[val_idx],
                'output_success': y_su[val_idx]
            }
        )).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    
    # 3. 模型构建与训练/加载
    is_trained = False
    model = None
    
    if os.path.exists(MODEL_SAVE_PATH):
        print(f"Step 3: 检测到已保存模型 {MODEL_SAVE_PATH}，尝试加载...")
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("✅ 模型加载成功，跳过训练。")
            is_trained = True
        except Exception as e:
            print(f"❌ 模型加载失败: {e}. 将重新构建并训练模型。")
            is_trained = False
            
    if not is_trained:
        # 构建模型
        model = build_context_model(LOOK_BACK, vocab_size, EMBEDDING_DIM)
        
        # 训练
        print(f"Step 4: 开始训练 (Epochs={EPOCHS}, Batch={BATCH_SIZE})...")
        model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
        
        # 训练完成后保存
        try:
            model.save(MODEL_SAVE_PATH)
            print(f"\n✅ 模型已成功保存到文件夹: {MODEL_SAVE_PATH}")
        except Exception as save_e:
            print(f"\n❌ 模型保存失败: {save_e}")
            
    # 4. 批量预测并获取结果 (显示进度条)
    val_inputs = {
        'input_history': X_s[val_idx],
        'input_difficulty': X_d[val_idx],
        'input_word_id': X_w[val_idx],
        'input_user_bias': X_b[val_idx]
    }
    val_labels = {
        'output_steps': y_st[val_idx],
        'output_success': y_su[val_idx]
    }

    pred_steps, pred_success_prob, mae = evaluate_model_and_get_preds(model, val_inputs, val_labels)

    # 计算 ACC
    true_wins = val_labels['output_success']
    pred_wins = (pred_success_prob >= 0.5).astype(int)
    acc = accuracy_score(true_wins, pred_wins)
    
    # 5. 验证抽样报告 (使用批量结果计算的 MAE/ACC)
    
    # 生成报告的头部和汇总指标
    report = f"""
========================================
  LSTM模型验证报告 (Validation Report)
========================================
1. 平均步数误差 (MAE)    : {mae:.4f}
2. 胜负预测准确率        : {acc:.3%}
3. 大型误差率 (>{LARGE_ERROR_THRESHOLD}步)  : {np.mean(np.abs(val_labels['output_steps'] - pred_steps) > LARGE_ERROR_THRESHOLD):.3%}
========================================
"""
    # 清空 output.txt 并写入全局报告
    with open("outputs/lstm_output.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    
    # 调用 perform_validation 进行抽样报告
    perform_validation(model, user_map, valid_users, LOOK_BACK, 
                       VALIDATION_SAMPLE_SIZE, LARGE_ERROR_THRESHOLD, pred_steps)

if __name__ == "__main__":
    main()