import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, Embedding, Flatten, Concatenate
from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, roc_auc_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from tensorflow.keras.mixed_precision import set_global_policy
from tensorflow.keras.callbacks import EarlyStopping
import random
import os
import io
import wandb
from wandb.keras import WandbCallback
import matplotlib.pyplot as plt

# WandB全局控制变量
WANDB_ENABLED = True  # 控制是否启用WandB
wandb_run = None  # 存储wandb run实例
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
LSTM_UNITS = 32      # LSTM隐藏单元数
LSTM_LAYERS = 2      # LSTM层数
DROPOUT_RATE = 0.3   # Dropout比率
EMBEDDING_DIM = 32   # 词嵌入维度

# 训练配置
EPOCHS = 10         
BATCH_SIZE = 2048    
LEARNING_RATE = 0.001 # 学习率
PATIENCE = 4         # 早停耐心值

# 评估配置
LARGE_ERROR_THRESHOLD = 2.0  # 修改为偏差2步

# 其他配置
MODEL_SAVE_PATH = 'LSTM_Model'

# WandB 配置
WANDB_PROJECT = 'wordle-prediction'
WANDB_RUN_NAME = 'lstm-experiment'

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
                
                # 标签 - 修改为分类任务
                # 确保target_trial在1-7范围内
                trial_category = min(target_trial, 7)
                # 创建one-hot编码 (7个类别，对应1-7步)
                one_hot = np.zeros(7, dtype=np.float32)
                one_hot[trial_category - 1] = 1.0  # 索引从0开始，步数从1开始
                y_steps.append(one_hot)
                y_success.append(1.0 if target_trial <= 6 else 0.0)
            
            valid_players.append(user)

    return (
        np.array(X_seq_list, dtype=np.float32),
        np.array(X_diff_list, dtype=np.float32),
        np.array(X_word_list, dtype=np.float32),
        np.array(X_bias_list, dtype=np.float32), # 新增输出
        np.array(y_steps),  # one-hot编码已为float32
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
    
    # 根据LSTM_LAYERS参数动态构建多层LSTM
    x1 = input_hist
    for i in range(LSTM_LAYERS):
        return_sequences = (i < LSTM_LAYERS - 1)  # 最后一层不返回序列
        x1 = LSTM(LSTM_UNITS if i > 0 else 128, return_sequences=return_sequences)(x1)
        x1 = Dropout(DROPOUT_RATE)(x1)
    
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
    # 修改为7分类任务（1-7步），使用softmax激活函数
    out_steps = Dense(7, activation='softmax', name='output_steps', dtype='float32')(Dense(32, activation='relu')(z))
    out_success = Dense(1, activation='sigmoid', name='output_success', dtype='float32')(Dense(16, activation='relu')(z))
    
    # 必须更新 inputs 列表
    model = Model(inputs=[input_hist, input_diff, input_word, input_bias], outputs=[out_steps, out_success])
    
    # 使用指定的学习率创建优化器
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(optimizer=optimizer,
                  loss={'output_steps': 'categorical_crossentropy', 'output_success': 'binary_crossentropy'},
                  loss_weights={'output_steps': 1.0, 'output_success': 0.5},
                  metrics={'output_steps': ['accuracy'], 'output_success': 'accuracy'})
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
    """进行批量预测，显示进度条，并返回预测结果。"""
    print("\nStep 4: 开始批量预测，显示进度条...")
    
    # 批量预测 (verbose=1 开启进度条)
    predictions = model.predict(val_inputs, batch_size=BATCH_SIZE, verbose=1)
    
    # 对于分类任务，predictions[0]是类别概率分布
    pred_steps_probs = predictions[0]
    pred_success_prob = predictions[1].flatten()
    
    # 从one-hot编码的真实标签中获取类别索引 (1-7)
    true_steps_discrete = np.argmax(val_labels['output_steps'], axis=1) + 1  # +1因为索引从0开始，步数从1开始
    
    # 获取预测的类别 (1-7)
    pred_steps_discrete = np.argmax(pred_steps_probs, axis=1) + 1
    
    return pred_steps_probs, pred_steps_discrete, true_steps_discrete, pred_success_prob

def perform_validation(model, user_history_map, valid_players, look_back, sample_size, threshold, pred_steps_full):
    print("🔍 开始执行perform_validation函数...")
    
    buffer = io.StringIO()
    
    eligible = [u for u in valid_players if len(user_history_map[u]) >= look_back + 1]
    if len(eligible) < sample_size: sample_size = len(eligible)
    print(f"🔍 符合条件的用户数量: {len(eligible)}, 抽样数量: {sample_size}")
    
    # 更新输出路径信息
    buffer.write(f"\nStep 5: 启动回测验证抽样报告 (样本数={sample_size}, 输出至 outputs/lstm_output.txt)...")
    
    header = f"{'User ID':<10} | {'Bias':<5} | {'Diff':<5} | {'Real':<5} | {'Pred':<5} | {'Err':<5} | {'Status'}"
    buffer.write("\n" + "-" * 70 + "\n" + header + "\n" + "-" * 70)
    
    # 确保有足够的用户进行抽样
    if len(eligible) > 0:
        report_users = random.sample(eligible, min(10, sample_size))
        print(f"🔍 已抽取 {len(report_users)} 个用户进行展示")
        
        large_errors = 0
        
        for i, user in enumerate(report_users):
            full_hist = user_history_map[user]
            # 获取目标数据 (Trial, Difficulty, WordID, UserBias)
            target_data = full_hist[-1]
            
            real_trial = float(target_data[0])
            t_diff = target_data[1]; t_word = target_data[2]; t_bias = target_data[3]
            
            try:
                # 临时进行单样本预测以获得该用户的预测值
                temp_inputs = prepare_single_inference_input(full_hist[-(look_back+1) : -1], t_diff, t_word, t_bias, look_back)
                pred_probs, _ = model.predict(temp_inputs, verbose=0)
                
                # 获取预测的类别 (1-7)
                pred_discrete = np.argmax(pred_probs[0]) + 1  # +1因为索引从0开始，步数从1开始
                
                # 计算类别误差
                err = abs(pred_discrete - real_trial)
                
                if err > threshold: large_errors += 1
                
                status = "✅" if err == 0 else "⚠️"
                if err > threshold: status = "❌"
                line = f"{str(user):<10} | {t_bias:.2f}  | {t_diff:.2f}  | {real_trial:.0f}    | {pred_discrete}      | {err}        | {status}"
                buffer.write(f"\n{line}")
            except Exception as e:
                print(f"❌ 处理用户 {user} 时出错: {e}")
                buffer.write(f"\n{str(user):<10} | 错误: {str(e)[:20]}...")

        if sample_size > len(report_users):
            buffer.write(f"\n... (其余 {sample_size - len(report_users)} 条省略)")
    else:
        buffer.write("\n⚠️ 没有找到符合条件的用户数据进行抽样")
        print("⚠️ 没有找到符合条件的用户数据进行抽样")
    
    # 打印到控制台
    report_content = buffer.getvalue()
    print(report_content)
    
    # 写入文件，确保文件存在
    try:
        os.makedirs('outputs', exist_ok=True)
        with open("outputs/lstm_output.txt", "a", encoding="utf-8") as f: # 追加到报告
            f.write(report_content)
        print("\n✅ 抽样报告已成功导出至 outputs/lstm_output.txt 文件。")
    except Exception as e:
        print(f"\n❌ 写入抽样报告失败: {e}")

def plot_confusion_matrix(true_labels, pred_labels, model_name, save_dir):
    """绘制并保存混淆矩阵"""
    # 生成混淆矩阵
    cm = confusion_matrix(true_labels, pred_labels)
    
    # 计算每个类别的准确率百分比
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    plt.imshow(cm_percent, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(f'{model_name} 混淆矩阵 (%)')
    plt.colorbar()
    
    # 设置标签
    classes = list(range(1, 8))
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # 在混淆矩阵中添加百分比文本
    fmt = '.1f'
    thresh = cm_percent.max() / 2.
    for i in range(cm_percent.shape[0]):
        for j in range(cm_percent.shape[1]):
            plt.text(j, i, format(cm_percent[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm_percent[i, j] > thresh else "black")
    
    plt.ylabel('真实步数')
    plt.xlabel('预测步数')
    plt.tight_layout()
    
    # 保存混淆矩阵
    save_path = os.path.join(save_dir, f'{model_name}_confusion_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 混淆矩阵已保存至: {save_path}")
    return save_path


# ==========================================
# 4. 主程序 (Main)
# ==========================================

def plot_loss_curve(history, model_name, save_dir):
    """绘制并保存Loss曲线和准确率曲线"""
    plt.figure(figsize=(14, 10))
    
    # 绘制总损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title(f'{model_name} 总损失曲线')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制步数预测损失曲线
    plt.subplot(2, 2, 2)
    plt.plot(history.history['output_steps_loss'], label='训练步数损失')
    plt.plot(history.history['val_output_steps_loss'], label='验证步数损失')
    plt.title(f'{model_name} 步数预测损失')
    plt.xlabel('Epoch')
    plt.ylabel('分类损失 (Categorical Crossentropy)')
    plt.legend()
    plt.grid(True)
    
    # 绘制步数预测准确率曲线
    plt.subplot(2, 2, 3)
    plt.plot(history.history['output_steps_accuracy'], label='训练步数准确率')
    plt.plot(history.history['val_output_steps_accuracy'], label='验证步数准确率')
    plt.title(f'{model_name} 步数预测准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    # 绘制成功预测准确率曲线
    plt.subplot(2, 2, 4)
    plt.plot(history.history['output_success_accuracy'], label='训练成功预测准确率')
    plt.plot(history.history['val_output_success_accuracy'], label='验证成功预测准确率')
    plt.title(f'{model_name} 成功预测准确率')
    plt.xlabel('Epoch')
    plt.ylabel('准确率')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_loss_curve.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Loss曲线已保存至: {save_path}")
    return save_path

def plot_prediction_trends(true_steps, pred_steps, model_name, save_dir):
    """绘制预测结果趋势图"""
    # 随机选择100个样本进行可视化，避免图过于拥挤
    if len(true_steps) > 100:
        indices = np.random.choice(len(true_steps), 100, replace=False)
        true_sample = true_steps[indices]
        pred_sample = pred_steps[indices]
    else:
        true_sample = true_steps
        pred_sample = pred_steps
    
    # 按真实值排序
    sorted_indices = np.argsort(true_sample)
    true_sample_sorted = true_sample[sorted_indices]
    pred_sample_sorted = pred_sample[sorted_indices]
    
    plt.figure(figsize=(12, 6))
    
    # 绘制预测值与真实值对比图
    plt.subplot(1, 2, 1)
    plt.scatter(true_sample, pred_sample, alpha=0.5, s=50)
    plt.plot([0, 7], [0, 7], 'r--', lw=2)  # 理想线
    plt.title(f'{model_name} 预测值 vs 真实值')
    plt.xlabel('真实步数')
    plt.ylabel('预测步数')
    plt.grid(True)
    plt.xlim(0, 7)
    plt.ylim(0, 7)
    
    # 绘制排序后的预测趋势
    plt.subplot(1, 2, 2)
    plt.plot(range(len(true_sample_sorted)), true_sample_sorted, 'b-', label='真实值')
    plt.plot(range(len(pred_sample_sorted)), pred_sample_sorted, 'r--', label='预测值')
    plt.title(f'{model_name} 预测趋势')
    plt.xlabel('样本索引 (按真实值排序)')
    plt.ylabel('步数')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_prediction_trends.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ 预测趋势图已保存至: {save_path}")
    return save_path

def train_model():
    """
    训练模型函数
    """
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
            print("✅ WandB 初始化成功（离线模式）")
        except Exception as e:
            print(f"❌ WandB 初始化失败: {str(e)}")
            print("ℹ️  程序将在不使用 WandB 的情况下继续运行")
            WANDB_ENABLED = False
            wandb_run = None
            config = {}
    else:
        print("ℹ️  WandB 已被禁用")
        config = {}
    config.look_back = LOOK_BACK
    config.epochs = EPOCHS
    config.batch_size = BATCH_SIZE
    config.embedding_dim = EMBEDDING_DIM
    config.large_error_threshold = LARGE_ERROR_THRESHOLD
        
    # 1. 数据处理
    user_map, vocab_size, _, _ = process_data_and_extract_features(FILE_PATH, MAX_ROWS)
    if not user_map: return None, None, None, None, None, None, None, None

    # 2. 数据集构建
    X_s, X_d, X_w, X_b, y_st, y_su, valid_users = create_multi_input_dataset(user_map, LOOK_BACK)
    
    # 划分训练/验证/测试集 = 7:1:2
    indices = np.arange(len(y_st))
    # 先划分训练集和剩余数据
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42)
    # 再从剩余数据中划分验证集和测试集
    val_idx, test_idx = train_test_split(temp_idx, test_size=2/3, random_state=42)  # 1:2
    
    print(f"数据集划分：训练集 {len(train_idx)}, 验证集 {len(val_idx)}, 测试集 {len(test_idx)}")
    
    return (X_s, X_d, X_w, X_b, y_st, y_su, valid_users, train_idx, val_idx, test_idx, 
            vocab_size, user_map, visualization_dir, config)
    
def build_and_train_model(X_s, X_d, X_w, X_b, y_st, y_su, train_idx, val_idx, test_idx, 
                          vocab_size, visualization_dir, config):
    """
    构建并训练模型
    """
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
    
    # 测试集也不需要shuffle
    test_ds = tf.data.Dataset.from_tensor_slices((
            {
                'input_history': X_s[test_idx],
                'input_difficulty': X_d[test_idx],
                'input_word_id': X_w[test_idx],
                'input_user_bias': X_b[test_idx]
            },
            {
                'output_steps': y_st[test_idx],
                'output_success': y_su[test_idx]
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
        
        # 创建早停回调
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # 训练
        print(f"Step 4: 开始训练 (Epochs={EPOCHS}, Batch={BATCH_SIZE})...")
          # 构建回调列表
          callbacks = [early_stopping]
          if WANDB_ENABLED and wandb_run is not None:
              callbacks.append(WandbCallback(save_model=False))
          
          history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1,
                            callbacks=callbacks)
            
        # 训练完成后保存
        try:
            model.save(MODEL_SAVE_PATH)
            print(f"\n✅ 模型已成功保存到文件夹: {MODEL_SAVE_PATH}")
        except Exception as save_e:
            print(f"\n❌ 模型保存失败: {save_e}")
        
        # 绘制并保存Loss曲线
        loss_curve_path = plot_loss_curve(history, 'LSTM', visualization_dir)
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
        
    return model, val_idx, test_idx
        
def evaluate_model(model, X_s, X_d, X_w, X_b, y_st, y_su, valid_users, user_map, 
                  val_idx, test_idx, visualization_dir, config):
    """
    评估模型在验证集和测试集上的表现
    """
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

    val_pred_probs, val_pred_discrete, val_true_discrete, val_pred_success_prob = evaluate_model_and_get_preds(model, val_inputs, val_labels)

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
    
    # 计算大型误差率
    val_large_error_rate = np.mean(np.abs(val_labels['output_steps'] - val_pred_steps) > LARGE_ERROR_THRESHOLD)
    
    # 计算离散预测的精确率、召回率、F1值
    val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
        val_true_discrete, val_pred_discrete, average='macro', zero_division=0
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

    test_pred_probs, test_pred_discrete, test_true_discrete, test_pred_success_prob = evaluate_model_and_get_preds(model, test_inputs, test_labels)

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
    
    # 计算测试集大型误差率
    test_large_error_rate = np.mean(np.abs(test_labels['output_steps'] - test_pred_steps) > LARGE_ERROR_THRESHOLD)
    
    # 计算测试集离散预测的精确率、召回率、F1值
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        test_true_discrete, test_pred_discrete, average='macro', zero_division=0
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
                "test_accuracy": test_acc,
                "test_auc": test_auc,
                "test_large_error_rate": test_large_error_rate,
                "test_precision": test_precision,
                "test_recall": test_recall,
                "test_f1": test_f1
            })
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
    
    # 生成报告的头部和汇总指标
    report = f"""
========================================
  LSTM模型验证和测试报告
========================================
---- 验证集指标 ----
3. 胜负预测准确率        : {val_acc:.3%}
4. ROC曲线下面积 (AUC)   : {val_auc:.4f}
5. 大型误差率 (>{LARGE_ERROR_THRESHOLD}步)  : {val_large_error_rate:.3%}
6. 精确率 (Precision)    : {val_precision:.4f}
7. 召回率 (Recall)       : {val_recall:.4f}
8. F1值 (F1-Score)       : {val_f1:.4f}

---- 测试集指标 ----
3. 胜负预测准确率        : {test_acc:.3%}
4. ROC曲线下面积 (AUC)   : {test_auc:.4f}
5. 大型误差率 (>{LARGE_ERROR_THRESHOLD}步)  : {test_large_error_rate:.3%}
6. 精确率 (Precision)    : {test_precision:.4f}
7. 召回率 (Recall)       : {test_recall:.4f}
8. F1值 (F1-Score)       : {test_f1:.4f}
========================================
"""
    # 清空 output.txt 并写入全局报告
    with open("outputs/lstm_output.txt", "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    
    # 调用 perform_validation 进行抽样报告（使用测试集预测结果）
    perform_validation(model, user_map, valid_users, LOOK_BACK, 
                       min(10000, len(test_idx)), LARGE_ERROR_THRESHOLD, test_pred_steps)
    
    # 绘制并保存预测趋势图（使用测试集结果）
    prediction_trend_path = plot_prediction_trends(
        test_labels['output_steps'], test_pred_steps, 'LSTM', visualization_dir
    )
    # 记录到WandB
    if WANDB_ENABLED and wandb_run is not None:
        try:
            wandb.log({'prediction_trends': wandb.Image(prediction_trend_path)})
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
    
    # 绘制并保存混淆矩阵（使用测试集离散化结果）
    cm_path = plot_confusion_matrix(test_true_discrete, test_pred_discrete, 'LSTM', visualization_dir)
    if WANDB_ENABLED and wandb_run is not None:
        try:
            wandb.log({'confusion_matrix': wandb.Image(cm_path)})
        except Exception as e:
            print(f"❌ WandB 日志记录失败: {str(e)}")
    
    return model

def predict_with_model(model, user_history_map, user_id, look_back):
    """
    使用训练好的模型进行预测
    
    参数:
    model: 训练好的模型
    user_history_map: 用户历史数据映射
    user_id: 要预测的用户ID
    look_back: 历史窗口大小
    
    返回:
    预测的步数、离散化的预测步数和成功概率
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
    
    # 计算离散化的预测值
    pred_continuous = min(pred_steps[0][0], 6.99)
    pred_discrete = int(round(pred_continuous))
    pred_discrete = max(1, min(7, pred_discrete))
    
    return pred_continuous, pred_discrete, pred_success[0][0]

def main(mode='train', user_id=None):
    """
    主函数
    
    参数:
    mode: 'train' 或 'predict'
    user_id: 当mode为'predict'时，指定要预测的用户ID
    """
    if mode == 'train':
        # 训练模式
        data = train_model()
        if data[0] is None:  # 数据加载失败
            return
        
        X_s, X_d, X_w, X_b, y_st, y_su, valid_users, train_idx, val_idx, test_idx, \
        vocab_size, user_map, visualization_dir, config = data
        
        model, val_idx, test_idx = build_and_train_model(
            X_s, X_d, X_w, X_b, y_st, y_su, train_idx, val_idx, test_idx,
            vocab_size, visualization_dir, config
        )
        
        evaluate_model(
            model, X_s, X_d, X_w, X_b, y_st, y_su, valid_users, user_map,
            val_idx, test_idx, visualization_dir, config
        )
        
        # 完成 WandB 实验记录
        wandb.finish()
        print("✅ WandB 实验记录已完成")
        print(f"✅ 所有可视化结果已保存至: {visualization_dir}")
        
    elif mode == 'predict':
        # 预测模式
        if not user_id:
            print("❌ 预测模式需要指定 user_id 参数")
            return
        
        # 加载模型
        if not os.path.exists(MODEL_SAVE_PATH):
            print(f"❌ 未找到模型文件: {MODEL_SAVE_PATH}")
            return
        
        try:
            model = tf.keras.models.load_model(MODEL_SAVE_PATH)
            print("✅ 模型加载成功")
            
            # 加载数据进行预测
            user_map, _, _, _ = process_data_and_extract_features(FILE_PATH, MAX_ROWS)
            if not user_map:
                print("❌ 数据加载失败")
                return
            
            # 进行预测
            pred_continuous, pred_discrete, pred_success = predict_with_model(model, user_map, user_id, LOOK_BACK)
            if pred_continuous is not None:
                print(f"\n用户 {user_id} 的预测结果:")
                print(f"预测步数(连续): {pred_continuous:.2f}")
                print(f"预测步数(离散): {pred_discrete}")
                print(f"成功概率: {pred_success:.2%}")
                print(f"预测结果: {'成功' if pred_continuous <= 6 else '失败'}")
                
        except Exception as e:
            print(f"❌ 预测过程中出错: {e}")
    else:
        print(f"❌ 未知模式: {mode}，请使用 'train' 或 'predict'")

if __name__ == "__main__":
    # 确保必要的文件夹存在
    os.makedirs('wandb', exist_ok=True)
    os.makedirs('visualization', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)
    main()