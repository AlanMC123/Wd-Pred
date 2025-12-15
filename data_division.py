import pandas as pd
import numpy as np
import os
import concurrent.futures 
import time
from sklearn.model_selection import StratifiedShuffleSplit  # 用于分层抽样
import random

# 固定随机种子以确保结果可复现
random.seed(42)
np.random.seed(42)

class MultiThreadedDataProcessor:
    def __init__(self, dataset_dir='dataset', max_workers=None):
        """
        初始化多线程数据处理器
        
        Args:
            dataset_dir: 数据集目录路径
            max_workers: 最大线程数，默认使用CPU核心数的2倍
        """
        self.dataset_dir = dataset_dir
        self.max_workers = max_workers or max(4, os.cpu_count() * 2)
        self.input_file = os.path.join(dataset_dir, 'cleaned_dataset.csv')
        self.train_file = os.path.join(dataset_dir, 'train_data.csv')
        self.val_file = os.path.join(dataset_dir, 'val_data.csv')
        self.test_file = os.path.join(dataset_dir, 'test_data.csv')
        self.difficulty_file = os.path.join(dataset_dir, 'difficulty.csv')
        self.player_data_file = os.path.join(dataset_dir, 'player_data.csv')
        
        # 确保目录存在
        if not os.path.exists(dataset_dir):
            os.makedirs(dataset_dir)
            print(f"创建目录: {dataset_dir}")
    
    def load_and_split_dataset(self, test_size=0.2, val_size=0.25):
        """
        加载数据集，创建 'success' 标志，并使用分层抽样进行三路分割。
        
        Args:
            test_size: 最终测试集占总数据的比例 (e.g., 20%).
            val_size: 验证集占剩余数据的比例 (e.g., 25% of 80% is 20%).
        """
        start_time = time.time()
        print(f"1. 加载和分割数据集 ({self.input_file})...")

        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"输入文件缺失: {self.input_file}")

        df = pd.read_csv(self.input_file)

        # 确保 Trial 列是整数类型
        df['Trial'] = pd.to_numeric(df['Trial'], errors='coerce').fillna(7).astype(int)
        df['success'] = (df['Trial'] <= 6).astype(int)
        
        # 记录原始数据中的比例
        original_success_rate = df['success'].mean()
        print(f"    原始数据成功率 (Trial<=6): {original_success_rate:.2%}")
        
        # 定义分层变量
        X = df.drop(columns=['success'])
        y = df['success']
        
        # 第一次分割: 训练集 (Train) vs. 临时集 (Temp: Val + Test)
        # 确保 y (success/failure) 的比例在 train 和 temp 中保持一致
        sss_temp = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
        
        for train_index, temp_index in sss_temp.split(X, y):
            train_data = df.iloc[train_index]
            temp_data = df.iloc[temp_index]
            
        print(f"    训练集大小: {len(train_data)}")
        
        # 第二次分割: 验证集 (Val) vs. 测试集 (Test)
        # Val 占 Temp 的 val_size 比例
        # 重新定义分层变量 for Temp set
        X_temp = temp_data.drop(columns=['success'])
        y_temp = temp_data['success']
        
        # 调整 val_size 比例，使其是 temp_data 的一部分
        # test_size=0.2, val_size=0.25 (占剩余 80% 的 25%，即总体的 20%)
        # 如果 test_size 和 val_size 设定为 0.2，那么 val_size 应该是 0.5 (temp_data 的一半)
        
        # 检查并确保 val_size / (1 - test_size) 不超过 1
        val_test_ratio = val_size / (1 - test_size)
        if val_test_ratio >= 1.0:
            print("警告: 验证集和测试集的比例设置不合理，将使用 50/50 划分。")
            val_test_ratio = 0.5
            
        sss_val_test = StratifiedShuffleSplit(
            n_splits=1, 
            test_size=val_test_ratio, 
            random_state=42
        )

        for val_index, test_index in sss_val_test.split(X_temp, y_temp):
            val_data = temp_data.iloc[val_index]
            test_data = temp_data.iloc[test_index]
        
        print(f"    验证集大小: {len(val_data)}")
        print(f"    测试集大小: {len(test_data)}")
        
        # 移除 'success' 辅助列
        train_data = train_data.drop(columns=['success'])
        val_data = val_data.drop(columns=['success'])
        test_data = test_data.drop(columns=['success'])

        train_data.to_csv(self.train_file, index=False)
        val_data.to_csv(self.val_file, index=False)
        test_data.to_csv(self.test_file, index=False)
        
        # 验证比例
        print("\n    验证分割集的成功率:")
        print(f"      训练集成功率: {(train_data['Trial'] <= 6).mean():.2%}")
        print(f"      验证集成功率: {(val_data['Trial'] <= 6).mean():.2%}")
        print(f"      测试集成功率: {(test_data['Trial'] <= 6).mean():.2%}")


        print(f"\n    耗时: {time.time() - start_time:.2f} 秒")
        return train_data, val_data, test_data
    
    def calculate_word_difficulty(self, train_data):
        """
        计算每个单词的平均猜测步数，并保存到difficulty.csv
        使用多线程CPU加速计算
        """
        print("\nStep 2: 计算单词难度...")
        start_time = time.time()
        
        # 按单词分组统计平均猜测步数
        word_groups = train_data.groupby('target')
        word_list = list(word_groups.groups.keys())
        total_words = len(word_list)
        
        # 定义处理单个单词组的函数
        def process_word(word):
            group = word_groups.get_group(word)
            avg_trial = group['Trial'].mean()
            total_attempts = len(group)
            failure_rate = len(group[group['Trial'] > 6]) / total_attempts if total_attempts > 0 else 0
            return {
                'word': word,
                'avg_trial': avg_trial,
                'total_attempts': total_attempts,
                'failure_rate': failure_rate
            }
        
        # 使用线程池并行处理
        word_results = []
        batch_size = max(1, min(100, total_words // (self.max_workers * 2)))
        
        print(f"  使用 {self.max_workers} 个线程进行并行计算...")
        print(f"  总单词数: {total_words}, 每批处理: {batch_size} 个单词")
        
        # 分批处理以减少线程创建开销
        for i in range(0, total_words, batch_size):
            batch = word_list[i:i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_word, batch))
                word_results.extend(results)
            
            # 显示进度
            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total_words:
                progress = min((i + batch_size) / total_words * 100, 100)
                print(f"    进度: {progress:.1f}% ({i + batch_size}/{total_words})")
        
        # 创建单词难度 DataFrame
        difficulty_df = pd.DataFrame(word_results)
        
        # 按平均猜测步数排序（步数越多难度越大）
        difficulty_df = difficulty_df.sort_values(by='avg_trial', ascending=False)
        
        # 保存到文件
        difficulty_df.to_csv(self.difficulty_file, index=False)
        
        print(f"  单词难度计算完成:")
        print(f"    处理单词数: {len(difficulty_df)}")
        print(f"    最困难单词: {difficulty_df.iloc[0]['word']} (平均步数: {difficulty_df.iloc[0]['avg_trial']:.2f})")
        print(f"    最简单单词: {difficulty_df.iloc[-1]['word']} (平均步数: {difficulty_df.iloc[-1]['avg_trial']:.2f})")
        print(f"    耗时: {time.time() - start_time:.2f} 秒")
        print(f"    结果保存至: {self.difficulty_file}")
    
    def calculate_player_stats(self, train_data):
        """
        计算每个用户的猜词水平，包括平均次数、失败率和步数方差
        使用多线程加速计算
        """
        print("\nStep 3: 计算用户猜词水平...")
        start_time = time.time()
        
        # 按用户分组
        user_groups = train_data.groupby('Username')
        user_list = list(user_groups.groups.keys())
        total_users = len(user_list)
        
        # 定义处理单个用户的函数
        def process_user(username):
            user_data = user_groups.get_group(username)
            trials = user_data['Trial'].values
            
            # 计算用户统计指标
            avg_trial = trials.mean()
            failure_rate = np.mean(trials > 6)  # 步数>6表示失败（数据集中步数为7表示失败）
            trial_variance = trials.var()
            total_games = len(user_data)
            success_count = len(user_data[user_data['Trial'] <= 6])
            
            return {
                'Username': username,
                'avg_trial': avg_trial,
                'failure_rate': failure_rate,
                'trial_variance': trial_variance,
                'total_games': total_games,
                'success_count': success_count,
                'success_rate': success_count / total_games if total_games > 0 else 0
            }
        
        # 使用线程池并行处理
        player_results = []
        batch_size = max(1, min(100, total_users // (self.max_workers * 2)))
        
        print(f"  使用 {self.max_workers} 个线程进行并行计算...")
        print(f"  总用户数: {total_users}, 每批处理: {batch_size} 个用户")
        
        # 分批处理
        for i in range(0, total_users, batch_size):
            batch = user_list[i:i + batch_size]
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                results = list(executor.map(process_user, batch))
                player_results.extend(results)
            
            # 显示进度
            if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total_users:
                progress = min((i + batch_size) / total_users * 100, 100)
                print(f"    进度: {progress:.1f}% ({i + batch_size}/{total_users})")
        
        # 创建用户统计DataFrame
        player_df = pd.DataFrame(player_results)
        
        # 按平均步数排序（步数越少水平越高）
        player_df = player_df.sort_values(by='avg_trial')
        
        # 保存到文件
        player_df.to_csv(self.player_data_file, index=False)
        
        print(f"  用户猜词水平计算完成:")
        print(f"    处理用户数: {len(player_df)}")
        print(f"    水平最高用户: {player_df.iloc[0]['Username']} (平均步数: {player_df.iloc[0]['avg_trial']:.2f})")
        print(f"    水平最低用户: {player_df.iloc[-1]['Username']} (平均步数: {player_df.iloc[-1]['avg_trial']:.2f})")
        print(f"    耗时: {time.time() - start_time:.2f} 秒")
        print(f"    结果保存至: {self.player_data_file}")
    
    def run_complete_pipeline(self):
        """
        运行完整的数据预处理流程
        """
        print("===== 多线程数据预处理启动 =====")
        total_start_time = time.time()
        
        try:
            # 1. 加载并分割数据集
            train_data = self.load_and_split_dataset()
            
            # 2. 计算单词难度
            self.calculate_word_difficulty(train_data)
            
            # 3. 计算用户水平
            self.calculate_player_stats(train_data)
            
            total_time = time.time() - total_start_time
            print(f"\n===== 数据预处理完成 =====")
            print(f"总耗时: {total_time:.2f} 秒")
            print(f"生成文件:")
            print(f"  - 训练集: {self.train_file}")
            print(f"  - 验证集: {self.val_file}")
            print(f"  - 测试集: {self.test_file}")
            print(f"  - 单词难度: {self.difficulty_file}")
            print(f"  - 用户水平: {self.player_data_file}")
            
        except Exception as e:
            print(f"\n错误: {str(e)}")
            raise

def main():
    """主函数，执行完整的预处理流程"""
    print("开始数据预处理流程...")
    
    # 创建并运行数据处理器
    processor = MultiThreadedDataProcessor()
    processor.run_complete_pipeline()
    print("所有预处理任务完成!")

# 程序主入口
if __name__ == "__main__":
    main()