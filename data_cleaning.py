import pandas as pd
import os

def clean_dataset():
    """
    清洗数据集，筛选参与8次以上游戏的用户
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
    users_with_8plus_games = user_game_counts[user_game_counts >= 8].index.tolist()
    
    # 筛选符合条件的用户数据
    cleaned_df = df[df[user_column].isin(users_with_8plus_games)]
    
    # 保存清洗后的数据集
    output_file = os.path.join(dataset_dir, 'cleaned_dataset.csv')
    cleaned_df.to_csv(output_file, index=False)
    
    # 输出统计信息
    print(f"原始数据总行数: {len(df)}")
    print(f"参与8次以上游戏的用户数: {len(users_with_8plus_games)}")
    print(f"清洗后数据总行数: {len(cleaned_df)}")
    print(f"清洗后的数据集已保存至: {output_file}")
    
    return output_file

if __name__ == "__main__":
    clean_dataset()