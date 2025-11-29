import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Embedding, Flatten, Concatenate, MultiHeadAttention, LayerNormalization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import set_global_policy
import random
import os
import io
import wandb
from wandb.keras import WandbCallback

# WandB全局控制变量
WANDB_ENABLED = True  # 控制是否启用WandB
wandb_run = None  # 存储wandb run实例
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ==========================================
# 0. 用户配置参数 (STRICT CONFIG)
# ==========================================
# 数据配置
FILE_PATH = 'dataset/cleaned_dataset.csv'
MAX_ROWS = 6923127
LOOK_BACK = 8       # 历史窗口大小

# 模型结构配置
# Transformer 核心参数
NUM_HEADS = 6      # 多头注意力的头数
KEY_DIM = 36       # 键和查询的维度
FF_DIM = 108        # 前馈网络的维度

# 模型复杂度参数
TRANSFORMER_LAYERS = 2  # Transformer层的数量
DROPOUT_RATE = 0.3   # Dropout比率
EMBEDDING_DIM = 64   # 词嵌入维度

# 训练配置
EPOCHS = 10         
BATCH_SIZE = 2048    
LEARNING_RATE = 0.001 # 学习率
PATIENCE = 4         # 早停耐心值

# 评估配置
LARGE_ERROR_THRESHOLD = 2.0  # 大型误差的阈值（步）

# 其他配置
MODEL_SAVE_PATH = 'Transformer_Model'

# WandB 配置
WANDB_PROJECT = 'wordle-prediction'
WANDB_RUN_NAME = 'transformer-experiment'
WANDB_ENABLED = True  # 控制是否启用WandB
wandb_run = None  # 存储wandb run实例

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
                
                # --- 1. 历史序列特征 (Transformer Input) ---
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
                
                # 标签：将步数转换为0-6的索引（对应1-7步），然后进行one-hot编码
                step_idx = min(int(target_trial) - 1, 6)  # 1-7步转换为0-6索引，确保7步对应索引6
                y_steps.append(step_idx)
                y_success.append(1.0 if target_trial <= 6 else 0.0)
            
            valid_players.append(user)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_diff_list, dtype=np.float32),
        np.array(X_word_list, dtype=np.float32),
        np.array(X_bias_list, dtype=np.float32), # 新增输出
        to_categorical(np.array(y_steps, dtype=np.int32), num_classes=7),  # 转换为one-hot编码
        np.array(y_success, dtype=np.float32),
        valid_players
    )

# ==========================================
# 2. Transformer 模型构建 (Multi-Input Four Branch)
# ==========================================

class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            Dense(ff_dim, activation="relu"), 
            Dense(embed_dim),
        ])
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

def build_transformer_model(look_back, vocab_size, embedding_dim, num_heads, key_dim, ff_dim, transformer_layers=1, dropout_rate=0.3, learning_rate=0.001):
    print(f"Step 3: 构建Transformer神经网络 (Vocab={vocab_size}, Layers={transformer_layers})...")
    
    # --- Input 1: 玩家历史 (Transformer) ---
    input_hist = Input(shape=(look_back, 2), name='input_history')
    # 添加位置编码（简化版本）
    positions = tf.range(start=0, limit=look_back, delta=1)
    pos_encoding = Embedding(input_dim=look_back, output_dim=2)(positions)
    pos_encoding = tf.expand_dims(pos_encoding, 0)
    x1 = input_hist + pos_encoding  # 添加位置编码
    
    # 应用多个Transformer层
    for i in range(transformer_layers):
        transformer_block = TransformerBlock(embed_dim=2, num_heads=num_heads, ff_dim=ff_dim, rate=dropout_rate)
        x1 = transformer_block(x1)
    
    x1 = tf.keras.layers.GlobalAveragePooling1D()(x1)  # 池化为固定长度向量
    x1 = Dropout(dropout_rate)(x1)
    
    # --- Input 2: 单词难度 (Dense) ---
    input_diff = Input(shape=(1,), name='input_difficulty'); 
    x2 = Dense(16, activation='relu')(input_diff)
    
    # --- Input 3: 单词 ID (Embedding) ---
    input_word = Input(shape=(1,), name='input_word_id'); 
    x3 = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=1)(input_word); 
    x3 = Flatten()(x3)
    
    # --- Input 4: 用户偏好 (Dense) ---
    input_bias = Input(shape=(1,), name='input_user_bias'); 
    x4 = Dense(16, activation='relu')(input_bias) # 新增
    
    # --- 融合层 (Concatenate) ---
    combined = Concatenate()([x1, x2, x3, x4]) # 融合四个分支
    
    z = Dense(64, activation='relu')(combined); 
    z = Dropout(dropout_rate)(z)
    
    # --- 输出层 ---
    # 将步数预测从回归改为分类任务：7个类别(1-7步)
    out_steps = Dense(7, activation='softmax', name='output_steps', dtype='float32')(Dense(32, activation='relu')(z))
    out_success = Dense(1, activation='sigmoid', name='output_success', dtype='float32')(Dense(16, activation='relu')(z))
    
    # 必须更新 inputs 列表
    model = Model(inputs=[input_hist, input_diff, input_word, input_bias], outputs=[out_steps, out_success])
    
    # 使用指定的学习率
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    
    model.compile(optimizer=optimizer,
                  loss={'output_steps': 'categorical_crossentropy', 'output_success': 'binary_crossentropy'},
                  loss_weights={'output_steps': 1.0, 'output_success': 0.5},
                  metrics={'output_success': 'accuracy', 'output_steps': 'accuracy'})
    return model

# ==========================================
# 3. 验证与回测逻辑
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
    """进行批量预测，显示进度条，并计算分类指标。"""
    print("\nStep 4: 开始批量预测，显示进度条...")
    
    # 批量预测 (verbose=1 开启进度条)
    predictions = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=1)
    
    # 处理分类输出：获取概率最高的类别索引
    pred_steps_probs = predictions[0]  # 形状为(batch_size, 7)的概率分布
    pred_steps_discrete = np.argmax(pred_steps_probs, axis=1) + 1  # 将0-6索引转换为1-7步
    
    # 生成连续预测值用于可视化（取概率加权平均值）
    step_values = np.arange(1, 8)  # [1,2,3,4,5,6,7]
    pred_steps_continuous = np.sum(pred_steps_probs * step_values, axis=1)
    
    pred_success_prob = predictions[1].flatten()
    
    return pred_steps_continuous, pred_steps_discrete, pred_success_prob

def perform_validation(model, user_history_map, valid_players, look_back, sample_size, threshold, pred_steps_full):
    
    buffer = io.StringIO()
    
    eligible = [u for u in valid_players if len(user_history_map[u]) >= look_back + 1]
    if len(eligible) < sample_size: sample_size = len(eligible)
    
    buffer.write(f"\nStep 5: 启动回测验证抽样报告 (样本数={sample_size}, 输出至 output.txt)...")
    
    header = f"{'User ID':<10} | {'Bias':<5} | {'Diff':<5} | {'Real':<5} | {'Pred_Cont':<8} | {'Pred_Disc':<5} | {'Err':<5} | {'Status'}"
    buffer.write("\n" + "-" * 75 + "\n" + header + "\n" + "-" * 75)
    
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
        p_steps_probs, _ = model.predict(temp_inputs, verbose=0)
        
        # 处理分类输出
        pred_val_disc = np.argmax(p_steps_probs[0]) + 1  # 将0-6索引转换为1-7步
        
        # 生成连续预测值（概率加权平均）
        step_values = np.arange(1, 8)  # [1,2,3,4,5,6,7]
        pred_val_cont = np.sum(p_steps_probs[0] * step_values)
        
        err = abs(pred_val_cont - real_trial)
        
        if err > threshold: large_errors += 1
        
        status = "✅" if err < 1.0 else "⚠️"
        if err > threshold: status = "❌"
        line = f"{str(user):<10} | {t_bias:.2f}  | {t_diff:.2f}  | {real_trial:.0f}    | {pred_val_cont:.2f}     | {pred_val_disc}      | {err:.2f}  | {status}"
        buffer.write(f"\n{line}")

    if sample_size > len(report_users):
        buffer.write(f"\n... (其余 {sample_size - len(report_users)} 条省略)")
    
    # 打印到控制台
    print(buffer.getvalue())
    
    # 写入文件
    with open("outputs/transformer_output.txt", "a", encoding="utf-8") as f: 
        f.write(buffer.getvalue())
    print("\n✅ 抽样报告已成功导出至 outputs/transformer_output.txt 文件。")

# ==========================================
# 4. 主程序 (Main)
# ==========================================

def plot_loss_curve(history, model_name, save_dir):
    """绘制并保存Loss曲线"""
    plt.figure(figsize=(12, 6))
    
    # 绘制总损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 总损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制步数预测损失曲线 - 分类任务使用交叉熵损失
    plt.subplot(1, 2, 2)
    plt.plot(history.history['output_steps_loss'], label='训练步数损失')
    plt.plot(history.history['val_output_steps_loss'], label='验证步数损失')
    plt.title(f'{model_name} 步数预测损失')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.grid(True)
    
    # 尝试添加步数预测的准确率曲线
    if 'output_steps_accuracy' in history.history and 'val_output_steps_accuracy' in history.history:
        plt.figure(figsize=(12, 6))
        plt.plot(history.history['output_steps_accuracy'], label='训练步数准确率')
        plt.plot(history.history['val_output_steps_accuracy'], label='验证步数准确率')
        plt.title(f'{model_name} 步数预测准确率')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 保存准确率曲线
        acc_path = os.path.join(save_dir, f'{model_name}_accuracy_curve.png')
        plt.savefig(acc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 准确率曲线已保存至: {acc_path}")
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss曲线已保存至: {save_path}")
    return save_path

def plot_prediction_trends(true_steps, pred_steps_cont, pred_steps_disc, model_name, save_dir):
    """绘制预测结果趋势图"""
    # 随机选择100个样本进行可视化，避免图过于拥挤
    if len(true_steps) > 100:
        indices = np.random.choice(len(true_steps), 100, replace=False)
        true_sample = true_steps[indices]
        pred_sample_cont = pred_steps_cont[indices]
        pred_sample_disc = pred_steps_disc[indices]
    else:
        true_sample = true_steps
        pred_sample_cont = pred_steps_cont
        pred_sample_disc = pred_steps_disc
    
    # 按真实值排序
    sorted_indices = np.argsort(true_sample)
    true_sample_sorted = true_sample[sorted_indices]
    pred_sample_cont_sorted = pred_sample_cont[sorted_indices]
    pred_sample_disc_sorted = pred_sample_disc[sorted_indices]
    
    plt.figure(figsize=(15, 6))
    
    # 绘制预测值与真实值对比图
    plt.subplot(1, 3, 1)
    plt.scatter(true_sample, pred_sample_cont, alpha=0.5, s=50, label='连续预测')
    plt.scatter(true_sample, pred_sample_disc, alpha=0.5, s=50, c='green', label='离散预测')
    plt.plot([0, 7], [0, 7], 'r--', lw=2)  # 理想线
    plt.title(f'{model_name} 预测值 vs 真实值')
    plt.xlabel('真实步数')
    plt.ylabel('预测步数')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    
    # 绘制排序后的预测趋势（连续值）
    plt.subplot(1, 3, 2)
    plt.plot(range(len(true_sample_sorted)), true_sample_sorted, 'b-', label='真实值')
    plt.plot(range(len(pred_sample_cont_sorted)), pred_sample_cont_sorted, 'r--', label='连续预测')
    plt.title(f'{model_name} 连续预测趋势')
    plt.xlabel('样本索引 (按真实值排序)')
    plt.ylabel('步数')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 7)
    
    # 绘制排序后的预测趋势（离散值）
    plt.subplot(1, 3, 3)
    plt.plot(range(len(true_sample_sorted)), true_sample_sorted, 'b-', label='真实值')
    plt.plot(range(len(pred_sample_disc_sorted)), pred_sample_disc_sorted, 'g--', label='离散预测 (1-7)')
    plt.title(f'{model_name} 离散预测趋势')
    plt.xlabel('样本索引 (按真实值排序)')
    plt.ylabel('步数')
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 7)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_prediction_trends.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测趋势图已保存至: {save_path}")
    return save_path

def plot_confusion_matrix(true_steps, pred_steps_disc, model_name, save_dir):
    """绘制混淆矩阵并保存"""
    # 确保真实值也转换为整数
    true_steps_disc = true_steps.astype(int)
    
    # 计算混淆矩阵
    cm = confusion_matrix(true_steps_disc, pred_steps_disc)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[str(i) for i in range(1, 8)],
                yticklabels=[str(i) for i in range(1, 8)])
    plt.title(f'{model_name} 预测步数混淆矩阵 (1-7)')
    plt.xlabel('预测步数')
    plt.ylabel('真实步数')
    plt.tight_layout()
    
    # 保存混淆矩阵
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")
    return save_path

def predict_with_model(model, user_history_map, user_id, look_back):
    """
    使用训练好的模型进行预测
    
    参数:
    model: 训练好的模型
    user_history_map: 用户历史数据映射
    user_id: 要预测的用户ID
    look_back: 历史窗口大小
    
    返回:
    连续预测步数、离散预测步数(1-7)和成功概率
    """
    if user_id not in user_history_map or len(user_history_map[user_id]) < look_back + 1:
        print(f"⚠️ 用户 {user_id} 数据不足，无法进行预测")
        return None, None, None
    
    full_hist = user_history_map[user_id]
    # 获取目标数据 (Trial, Difficulty, WordID, UserBias)
    target_data = full_hist[-1]
    t_diff = target_data[1]; t_word = target_data[2]; t_bias = target_data[3]
    
    # 准备输入数据
    inputs = prepare_single_inference_input(full_hist[-(look_back+1) : -1], t_diff, t_word, t_bias, look_back)
    pred_steps, pred_success = model.predict(inputs, verbose=0)
    
    # 连续预测值
    pred_steps_cont = min(pred_steps[0][0], 6.99)
    # 离散预测值（1-7整数）
    pred_steps_disc = int(round(pred_steps_cont))
    pred_steps_disc = max(1, min(7, pred_steps_disc))  # 确保在1-7范围内
    
    return pred_steps_cont, pred_steps_disc, pred_success[0][0]

def main(mode='train', user_id=None):
    """
    主函数
    
    参数:
    mode: 'train' 或 'predict'
    user_id: 当mode为'predict'时，指定要预测的用户ID
    """
    global WANDB_ENABLED, wandb_run
    if mode == 'train':
        # --- GPU 设置 ---
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus: tf.config.experimental.set_memory_growth(gpu, True)
                set_global_policy('mixed_float16')
                print("✅ GPU 加速已开启 (Mixed Float16)")
            except: pass
        
        # 确保可视化文件夹存在
        visualization_dir = 'visualization'
        os.makedirs(visualization_dir, exist_ok=True)
        os.makedirs('outputs', exist_ok=True)
        
        # --- 初始化 WandB ---
        global wandb_run
        if WANDB_ENABLED:
            print("🔄 初始化 WandB 实验记录...")
            try:
                # 使用离线模式避免网络连接问题
                wandb_run = wandb.init(
                    project=WANDB_PROJECT, 
                    name=WANDB_RUN_NAME, 
                    dir='wandb', 
                    anonymous='must',
                    resume=False,
                    mode='offline',  # 添加离线模式
                    settings=wandb.Settings(
                        start_method='thread',
                        disable_git=True,
                        disable_code=True
                    )
                )
                # 记录超参数
                config = wandb_run.config
                config.look_back = LOOK_BACK
                config.epochs = EPOCHS
                config.batch_size = BATCH_SIZE
                config.embedding_dim = EMBEDDING_DIM
                config.num_heads = NUM_HEADS
                config.ff_dim = FF_DIM
                config.transformer_layers = TRANSFORMER_LAYERS
                config.dropout_rate = DROPOUT_RATE
                config.learning_rate = LEARNING_RATE
                config.patience = PATIENCE
                config.large_error_threshold = LARGE_ERROR_THRESHOLD
                print("✅ WandB 初始化成功（离线模式）")
            except Exception as e:
                print(f"❌ WandB 初始化失败: {str(e)}")
                print("ℹ️  程序将在不使用 WandB 的情况下继续运行")
                WANDB_ENABLED = False
                wandb_run = None
        else:
            print("ℹ️  WandB 已被禁用")
        
    # 确保outputs目录存在
    os.makedirs('outputs', exist_ok=True)
    
    # 1. 数据处理
    user_map, vocab_size, _, _ = process_data_and_extract_features(FILE_PATH, MAX_ROWS)
    if not user_map: return 

    # 2. 数据集构建
    X_s, X_d, X_w, X_b, y_st, y_su, valid_users = create_multi_input_dataset(user_map, LOOK_BACK)
    
    # 划分训练/验证/测试集 = 7:1:2
    indices = np.arange(len(y_st))
    # 先划分训练集和剩余数据
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    # 再从剩余数据中划分验证集和测试集
    val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, random_state=42)  # 1:2
    
    print(f"数据集划分：训练集 {len(train_idx)}, 验证集 {len(val_idx)}, 测试集 {len(test_idx)}")
    
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
        # 构建Transformer模型
        model = build_transformer_model(LOOK_BACK, vocab_size, EMBEDDING_DIM, NUM_HEADS, KEY_DIM, FF_DIM, 
                                       transformer_layers=TRANSFORMER_LAYERS, 
                                       dropout_rate=DROPOUT_RATE, 
                                       learning_rate=LEARNING_RATE)
        
        # 设置早停回调
        early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
        
        # 训练
        print(f"Step 4: 开始训练 (Epochs={EPOCHS}, Batch={BATCH_SIZE}, Patience={PATIENCE})...")
        # 根据WandB是否启用决定回调列表
        callbacks = [early_stopping]
        if WANDB_ENABLED and wandb_run is not None:
            callbacks.append(WandbCallback(save_model=False))
        
        history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1,
                           callbacks=callbacks)
        
        # 绘制并保存Loss曲线
        loss_curve_path = plot_loss_curve(history, 'Transformer', visualization_dir)
        # 记录到WandB
        if WANDB_ENABLED and wandb_run is not None:
            try:
                wandb.log({'loss_curve': wandb.Image(loss_curve_path)})
            except Exception as e:
                print(f"❌ WandB 日志记录失败: {str(e)}")
        
        # 训练完成后保存
        try:
            model.save(MODEL_SAVE_PATH)
            print(f"\n✅ 模型已成功保存到文件夹: {MODEL_SAVE_PATH}")
        except Exception as save_e:
            print(f"\n❌ 模型保存失败: {save_e}")
            
    # 验证集评估
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

    val_pred_steps_cont, val_pred_steps_disc, val_pred_success_prob = evaluate_model_and_get_preds(model, val_inputs, val_labels)

    # 计算 ACC 和 AUC
    val_true_wins = val_labels['output_success']
    val_pred_wins = (val_pred_success_prob >= 0.5).astype(int)
    val_acc = accuracy_score(val_true_wins, val_pred_wins)
    
    # 计算 AUC
    try:
        val_auc = roc_auc_score(val_true_wins, val_pred_success_prob)
    except ValueError:
        print("⚠️ AUC 计算失败，可能是因为正负样本不平衡或仅有单一类别")
        val_auc = 0.5  # 默认值
    
    # 将 one-hot 标签转换为离散步数 (1-7)
    val_true_steps = np.argmax(val_labels['output_steps'], axis=1) + 1
    # 计算大型误差率
    val_large_error_rate = np.mean(np.abs(val_true_steps - val_pred_steps_cont) > LARGE_ERROR_THRESHOLD)
    
    # FIX: 将 one-hot 标签 (N, 7) 转换为离散步数 (N,)
    # np.argmax 找到 1 的索引 (0-6)，+1 转换为步数 (1-7)
    val_true_steps_disc = np.argmax(val_labels['output_steps'], axis=1) + 1
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_true_steps_disc, val_pred_steps_disc, average='weighted', zero_division=0
    )
    
    # 测试集评估
    test_inputs = {
        'input_history': X_s[test_idx],
        'input_difficulty': X_d[test_idx],
        'input_word_id': X_w[test_idx],
        'input_user_bias': X_b[test_idx]
    }
    test_labels = {
        'output_steps': y_st[test_idx],
        'output_success': y_su[test_idx]
    }

    test_pred_steps_cont, test_pred_steps_disc, test_pred_success_prob = evaluate_model_and_get_preds(model, test_inputs, test_labels)

    # 计算测试集 ACC 和 AUC
    test_true_wins = test_labels['output_success']
    test_pred_wins = (test_pred_success_prob >= 0.5).astype(int)
    test_acc = accuracy_score(test_true_wins, test_pred_wins)
    
    # 计算测试集 AUC
    try:
        test_auc = roc_auc_score(test_true_wins, test_pred_success_prob)
    except ValueError:
        print("⚠️ 测试集 AUC 计算失败，可能是因为正负样本不平衡或仅有单一类别")
        test_auc = 0.5  # 默认值
    
    # 将 one-hot 标签转换为离散步数 (1-7)
    test_true_steps = np.argmax(test_labels['output_steps'], axis=1) + 1
    # 计算大型误差率
    test_large_error_rate = np.mean(np.abs(test_true_steps - test_pred_steps_cont) > LARGE_ERROR_THRESHOLD)
    
    # FIX: 将 one-hot 标签 (N, 7) 转换为离散步数 (N,)
    test_true_steps_disc = np.argmax(test_labels['output_steps'], axis=1) + 1
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_true_steps_disc, test_pred_steps_disc, average='weighted', zero_division=0
    )
    # 记录验证指标到 WandB
    if WANDB_ENABLED and wandb_run is not None:
        try:
            wandb.log({
                "val_accuracy": val_acc,
                "val_auc": val_auc,
                "val_large_error_rate": val_large_error_rate,
                "val_precision": val_precision,
                "val_recall": val_recall,
                "val_f1": val_f1,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1,
                "test_accuracy": test_acc,
                "test_auc": test_auc,
                "test_large_error_rate": test_large_error_rate
            })
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
    
    # 5. 验证抽样报告 (使用批量结果计算的 MAE/ACC)
    
    # 生成报告的头部和汇总指标
    report = f"""
========================================
  Transformer模型验证和测试报告
========================================
---- 验证集指标 ----
1. 胜负预测准确率        : {val_acc:.3%}
2. ROC曲线下面积 (AUC)   : {val_auc:.4f}
3. 大型误差率 (>{LARGE_ERROR_THRESHOLD}步)  : {val_large_error_rate:.3%}
4. 精确率 (Precision)    : {val_precision:.4f}
5. 召回率 (Recall)       : {val_recall:.4f}
6. F1值 (F1-Score)       : {val_f1:.4f}

---- 测试集指标 ----
1. 胜负预测准确率        : {test_acc:.3%}
2. ROC曲线下面积 (AUC)   : {test_auc:.4f}
3. 大型误差率 (>{LARGE_ERROR_THRESHOLD}步)  : {test_large_error_rate:.3%}
4. 精确率 (Precision)    : {test_precision:.4f}
5. 召回率 (Recall)       : {test_recall:.4f}
6. F1值 (F1-Score)       : {test_f1:.4f}
========================================
"""
    # 写入全局报告
    with open("outputs/transformer_output.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    
    # 调用 perform_validation 进行抽样报告（使用测试集预测结果）
    # 由于已划分测试集，不再需要VALIDATION_SAMPLE_SIZE参数
    perform_validation(model, user_map, valid_users, LOOK_BACK, 
                       min(10000, len(test_idx)), LARGE_ERROR_THRESHOLD, test_pred_steps_cont)
    
    # 绘制并保存预测趋势图（使用测试集结果）
    # FIX: 使用已转换为单值形式 (N,) 的真实步数 test_true_steps_disc 进行绘图
    prediction_trend_path = plot_prediction_trends(
        test_true_steps_disc, test_pred_steps_cont, test_pred_steps_disc, 'Transformer', visualization_dir
    )
    # 记录到WandB
    if WANDB_ENABLED and wandb_run is not None:
        try:
            wandb.log({'prediction_trends': wandb.Image(prediction_trend_path)})
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
    
    # 绘制并保存混淆矩阵（使用测试集结果）
    confusion_matrix_path = plot_confusion_matrix(
        test_labels['output_steps'], test_pred_steps_disc, 'Transformer', visualization_dir
    )
    # 记录到WandB
    if WANDB_ENABLED and wandb_run is not None:
        try:
            wandb.log({'confusion_matrix': wandb.Image(confusion_matrix_path)})
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
            # 进行预测
    pred_steps_cont, pred_steps_disc, pred_success = predict_with_model(model, user_map, user_id, LOOK_BACK)
    if pred_steps_cont is not None:
        print(f"\n用户 {user_id} 的预测结果:")
        print(f"连续预测步数: {pred_steps_cont:.2f}")
        print(f"离散预测步数 (1-7): {pred_steps_disc}")
        print(f"成功概率: {pred_success:.2%}")
        print(f"预测结果: {'成功' if pred_steps_disc <= 6 else '失败'}")
        
    else:
        print(f"❌ 未知模式: {mode}，请使用 'train' 或 'predict'")

if __name__ == "__main__":
    # 确保必要的文件夹存在
    os.makedirs('wandb', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    # 支持命令行参数
    import sys
    if len(sys.argv) > 1:
        mode = sys.argv[1]
        user_id = sys.argv[2] if len(sys.argv) > 2 else None
        main(mode, user_id)
    else:
        main()