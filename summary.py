import pandas as pd

FILE_PATH = 'dataset/cleaned_dataset.csv'

def summarize_data():
    """执行数据汇总分析的主要函数"""
    # 加载数据
    df = pd.read_csv(FILE_PATH)

    # 查看数据基本信息
    print('数据基本信息：')
    print(df.info())

    # 查看前5行数据
    print('\n前5行数据：')
    print(df.head())

    # 查看数据统计信息
    print('\n数据统计信息：')
    print(df.describe())

    # 检查Game列的唯一值
    print('\nGame列的唯一值数量：', df['Game'].nunique())

    # 检查target列的唯一值
    print('\nTarget列的唯一值数量：', df['target'].nunique())

    # 检查processed_text列的结构
    print('\nprocessed_text列的前5行：')
    print(df['processed_text'].head())

    # 检查processed_text列的长度分布
    print('\nprocessed_text列的长度统计：')

    # 检查Trial列的分布
    print('\nTrial列的分布：')
    print(df['Trial'].value_counts())

    # 检查Username列的分布
    print('\nUsername列的唯一值数量：', df['Username'].nunique())

    # 查看一个完整的游戏记录示例
    print('\n一个完整的游戏记录示例：')
    # 选择一个有多个尝试的游戏
    multi_trial_game = df[df['Trial'] > 1]['Game'].iloc[0]
    print(df[df['Game'] == multi_trial_game])

# 当脚本直接运行时执行汇总分析
if __name__ == '__main__':
    summarize_data()