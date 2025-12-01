import pandas as pd
import numpy as np
import os
import concurrent.futures
import time
# 移除 train_test_split，因为我们将使用自定义的基于时间的切分逻辑
# from sklearn.model_selection import train_test_split 
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
    
    def load_and_split_dataset(self):
        """
        加载并分割数据集为训练集、验证集和测试集 (按时间序列切分)
        比例为7:1:2，确保训练集数据时间上早于测试集数据
        """
        print("Step 1: 加载并按时间顺序切分数据集 (Train: 70%, Val: 10%, Test: 20%)...")
        start_time = time.time()
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"找不到清理后的数据集: {self.input_file}")
        
        # 读取数据集
        df = pd.read_csv(self.input_file)
        print(f"  原始数据总行数: {len(df)}")
        
        # 1. 关键步骤：按用户分组并按游戏编号（时间顺序）排序
        # 确保每个用户内部的记录是按时间递增的
        df = df.sort_values(by=['Username', 'Game']).reset_index(drop=True)
        
        train_list = []
        val_list = []
        test_list = []
        
        # 2. 对每个用户进行时间序列切分
        for username, group in df.groupby('Username'):
            total_count = len(group)
            
            # 计算切分点
            train_end = int(total_count * 0.7)
            val_end = train_end + int(total_count * 0.1) 
            
            # 为了确保所有记录都被分配，最后的测试集部分取剩余的所有记录
            # 这样实际比例会非常接近 7:1:2，且不会丢失任何数据
            
            # 切分：由于group已经按时间排序，按索引切分即为按时间切分
            train_part = group.iloc[:train_end]
            val_part = group.iloc[train_end:val_end]
            test_part = group.iloc[val_end:]
            
            train_list.append(train_part)
            val_list.append(val_part)
            test_list.append(test_part)
            
        # 3. 合并所有用户的切分结果
        train_data = pd.concat(train_list)
        val_data = pd.concat(val_list)
        test_data = pd.concat(test_list)
        
        # 4. 保存分割后的数据集
        train_data.to_csv(self.train_file, index=False)
        val_data.to_csv(self.val_file, index=False)
        test_data.to_csv(self.test_file, index=False)
        
        # 验证比例
        total_len = len(df)
        print(f"  数据集分割完成 (按时间顺序):")
        print(f"    训练集: {len(train_data)} 条记录 ({len(train_data)/total_len*100:.1f}%)")
        print(f"    验证集: {len(val_data)} 条记录 ({len(val_data)/total_len*100:.1f}%)")
        print(f"    测试集: {len(test_data)} 条记录 ({len(test_data)/total_len*100:.1f}%)")
        print(f"  耗时: {time.time() - start_time:.2f} 秒")
        
        return train_data, val_data, test_data
    
    # 以下方法 (calculate_word_difficulty 和 calculate_player_stats) 保持不变
    def calculate_word_difficulty(self, train_data):
        """
        计算每个单词的平均猜测步数，并保存到difficulty.csv
        使用多线程加速计算
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
        
        # 创建单词难度DataFrame
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
        
        return difficulty_df
    
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
            failure_rate = np.mean(trials > 6)  # 步数>6表示失败
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
        
        return player_df
    
    def run_complete_pipeline(self):
        """
        运行完整的数据预处理流程
        """
        print("===== 多线程数据预处理启动 =====")
        total_start_time = time.time()
        
        try:
            # 1. 加载并分割数据集
            train_data, val_data, test_data = self.load_and_split_dataset()
            
            # 2. 计算单词难度
            difficulty_df = self.calculate_word_difficulty(train_data)
            
            # 3. 计算用户水平
            player_df = self.calculate_player_stats(train_data)
            
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