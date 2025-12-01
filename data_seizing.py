import pandas as pd
import os
import random
import numpy as np

def clean_dataset(percentage):
    """
    清洗数据集，筛选参与5次以上游戏的用户
    """
    # 确保dataset目录存在
    dataset_dir = 'dataset'
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)
        print(f"创建目录: {dataset_dir}")
    
    # 读取原始数据集
    input_file = os.path.join(dataset_dir, 'wordle_games.csv')
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"找不到输入文件: {input_file}")
    
    print(f"正在读取数据集: {input_file}")
    df = pd.read_csv(input_file)
    
    # 调试：打印数据集的列名
    print("数据集的实际列名:")
    print(list(df.columns))
    
    # 确认使用'Username'列作为用户标识
    user_column = 'Username'
    
    # 检查必要列是否存在
    if user_column not in df.columns:
        print(f"错误：找不到用户列 '{user_column}'。数据集列名: {df.columns.tolist()}")
        return None
    
    print(f"使用列 '{user_column}' 作为用户ID")
    user_game_counts = df[user_column].value_counts()
    users_with_8plus_games = user_game_counts[user_game_counts >= 5].index.tolist()
    
    # 筛选符合条件的用户数据
    cleaned_df = df[df[user_column].isin(users_with_8plus_games)]
    
    # 随机选取指定比例的用户
    print(f"正在随机选取{percentage}比例的用户...")
    # 获取唯一用户列表
    unique_users = cleaned_df[user_column].unique()
    # 计算需要选取的用户数量
    sample_size = max(1, int(len(unique_users) * percentage)) 
    
    # 使用固定种子101确保结果可重现
    random.seed(101)
    print("使用固定种子101进行随机选取...")
    
    # 随机选取用户
    sampled_users = random.sample(list(unique_users), sample_size)
    # 只保留选中用户的数据
    sampled_df = cleaned_df[cleaned_df[user_column].isin(sampled_users)]
    
    # 清洗数据：检查缺失值、Trial范围和target单词长度
    print("开始清洗选取的子集数据...")
    
    # 检查并移除缺失值
    initial_rows = len(sampled_df)
    sampled_df = sampled_df.dropna()
    missing_values_removed = initial_rows - len(sampled_df)
    if missing_values_removed > 0:
        print(f"已移除 {missing_values_removed} 行缺失值数据")
    
    # 检查Trial是否为1-7的整数
    if 'Trial' in sampled_df.columns:
        before_trial_filter = len(sampled_df)
        sampled_df = sampled_df[sampled_df['Trial'].between(1, 7, inclusive='both')]
        sampled_df = sampled_df[sampled_df['Trial'].astype(str).str.isdigit()]
        trial_filtered = before_trial_filter - len(sampled_df)
        if trial_filtered > 0:
            print(f"已移除 {trial_filtered} 行Trial值不在1-7范围内的数据")
    else:
        print("警告：数据集中未找到'Trial'列")
    
    # 检查target是否为5个字母的单词
    if 'target' in sampled_df.columns:
        before_target_filter = len(sampled_df)
        # 确保target是字符串类型并检查长度为5
        sampled_df = sampled_df[sampled_df['target'].astype(str).str.match(r'^[a-zA-Z]{5}$')]
        target_filtered = before_target_filter - len(sampled_df)
        if target_filtered > 0:
            print(f"已移除 {target_filtered} 行target不是5个字母单词的数据")
    else:
        print("警告：数据集中未找到'target'列")
    
    print(f"清洗后的数据行数: {len(sampled_df)}")
    
    # 保存清洗后的数据集
    output_file = os.path.join(dataset_dir, 'cleaned_dataset.csv')
    sampled_df.to_csv(output_file, index=False)
    
    # 输出统计信息
    print(f"原始数据总行数: {len(df)}")
    print(f"参与8次以上游戏的用户数: {len(users_with_8plus_games)}")
    print(f"符合条件的唯一用户数: {len(unique_users)}")
    print(f"随机选取的用户比例: {percentage}")
    print(f"随机选取的用户数: {len(sampled_users)}")
    print(f"采样后数据总行数: {len(sampled_df)}")
    print(f"清洗后的数据集已保存至: {output_file}")
    
    return output_file

if __name__ == "__main__":
    clean_dataset(1/20)