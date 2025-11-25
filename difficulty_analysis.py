import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import csv

FILE_PATH = 'F:\Codes\wordle_games.csv'

# 设置字体为微软雅黑
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 创建difficulty文件夹
if not os.path.exists('difficulty'):
    os.makedirs('difficulty')

# 读取CSV文件
print("正在读取wordle_games.csv文件...")
df = pd.read_csv(FILE_PATH)

# 1. 输出尝试次数(Trial)直方图
print("正在生成尝试次数直方图...")
plt.figure(figsize=(12, 6))
plt.hist(df['Trial'], bins=range(1, df['Trial'].max() + 2), edgecolor='black', align='left')
plt.title('Wordle游戏尝试次数直方图')
plt.xlabel('尝试次数')
plt.ylabel('游戏次数(万)')
plt.xticks(range(1, df['Trial'].max() + 1))
# 设置y轴单位为万
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x/10000)))
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()

# 保存直方图
histogram_path = os.path.join('difficulty', 'trial_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"尝试次数直方图已保存到: {histogram_path}")

# 2. 计算每个目标词的平均尝试次数和失败率
print("正在计算每个目标词的平均尝试次数和失败率...")

# 计算平均尝试次数
target_avg_trial = df.groupby('target')['Trial'].mean().reset_index()
target_avg_trial.columns = ['target', 'avg_trial']

# 计算总游戏次数
total_games = df.groupby('target').size().reset_index(name='total_games')

# 计算失败次数（尝试次数>=6次且未猜中，假设Trial为6表示失败）
failed_games = df[df['Trial'] >= 6].groupby('target').size().reset_index(name='failed_games')

# 合并数据
word_stats = pd.merge(target_avg_trial, total_games, on='target')
word_stats = pd.merge(word_stats, failed_games, on='target', how='left')

# 填充缺失值（没有失败记录的单词）
word_stats['failed_games'] = word_stats['failed_games'].fillna(0)

# 计算失败率
word_stats['failure_rate'] = (word_stats['failed_games'] / word_stats['total_games'] * 100).round(2)

# 3. 设置n和m的值
n = 10  # 平均次数最多的n个词
m = 10  # 平均次数最少的m个词

# 4. 找出平均次数最多的n个词
most_difficult = word_stats.sort_values(by='avg_trial', ascending=False).head(n)

# 5. 找出平均次数最少的m个词
least_difficult = word_stats.sort_values(by='avg_trial', ascending=True).head(m)

# 6. 准备所有单词的数据，按平均尝试次数降序排序
all_words_sorted = word_stats.sort_values(by='avg_trial', ascending=False).reset_index(drop=True)

# 6. 保存结果到txt文件
result_path = os.path.join('difficulty', 'word_difficulty.txt')
with open(result_path, 'w', encoding='utf-8') as f:
    f.write("Wordle游戏难度分析结果\n")
    f.write("=" * 40 + "\n")
    f.write(f"平均尝试次数最多的{n}个词(从难到易):\n")
    f.write("-" * 40 + "\n")
    for index, row in most_difficult.iterrows():
        f.write(f"{row['target']}: {row['avg_trial']:.2f}次, 失败率: {row['failure_rate']:.2f}%\n")
    f.write("\n")
    f.write(f"平均尝试次数最少的{m}个词(从易到难):\n")
    f.write("-" * 40 + "\n")
    for index, row in least_difficult.iterrows():
        f.write(f"{row['target']}: {row['avg_trial']:.2f}次, 失败率: {row['failure_rate']:.2f}%\n")

# 7. 生成CSV文件，包含所有单词的排名、单词、平均尝试次数和失败率
print("正在生成CSV文件...")
csv_path = os.path.join('difficulty', 'word_difficulty_ranking.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    # 创建CSV写入器
    csv_writer = csv.writer(csvfile)
    
    # 写入表头
    csv_writer.writerow(['排名', '单词', '平均尝试次数', '失败率(%)'])
    
    # 写入数据，排名从1开始
    for idx, row in all_words_sorted.iterrows():
        ranking = idx + 1  # 排名从1开始
        csv_writer.writerow([ranking, row['target'], round(row['avg_trial'], 2), row['failure_rate']])

print(f"难度分析结果已保存到: {result_path}")
print(f"CSV排名文件已保存到: {csv_path}")
print("所有分析任务已完成!")