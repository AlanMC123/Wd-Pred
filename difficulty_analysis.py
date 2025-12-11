import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import csv

FILE_PATH = 'dataset/cleaned_dataset.csv'

# 设置字体为 Arial
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# 确保 difficulty 文件夹存在
if not os.path.exists('difficulty'):
    os.makedirs('difficulty')

# 读取数据集
print("Reading wordle_games.csv file...")
df = pd.read_csv(FILE_PATH)

# 1. 绘制尝试次数分布直方图
print("Generating try distribution histogram...")
plt.figure(figsize=(12, 6))
plt.hist(df['Trial'], bins=range(1, df['Trial'].max() + 2), edgecolor='black', align='left')
plt.title('Wordle Try Distribution')
plt.xlabel('Number of Tries')
plt.ylabel('Number of Games (10,000)')
plt.xticks(range(1, df['Trial'].max() + 1))
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x/10000)))
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()

# 保存直方图到文件
histogram_path = os.path.join('difficulty', 'try_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Try distribution histogram saved to: {histogram_path}")

# 2. 计算每个目标词的平均尝试次数和失败率
print("Calculating average Tries and failure rate for each target word...")

# 计算每个单词的平均尝试次数
target_avg_try = df.groupby('target')['Trial'].mean().reset_index()
target_avg_try.columns = ['target', 'avg_try']

# 计算每个单词的总游戏次数
total_games = df.groupby('target').size().reset_index(name='total_games')

# 计算每个单词的失败游戏次数 (假设 Trial = 7 表示失败)
failed_games = df[df['Trial'] == 7].groupby('target').size().reset_index(name='failed_games')

# 将结果合并到一个 DataFrame 中
word_stats = pd.merge(target_avg_try, total_games, on='target')
word_stats = pd.merge(word_stats, failed_games, on='target', how='left')

# 填补缺失值 (失败游戏次数为0)
word_stats['failed_games'] = word_stats['failed_games'].fillna(0)

# 计算每个单词的失败率
word_stats['failure_rate'] = (word_stats['failed_games'] / word_stats['total_games'] * 100).round(2)

# 3. 设置 n 和 m 的值，计算前10难和前10简单的词
n = 10  # 高难度单词数量
m = 10  # 低难度单词数量

# 4. 高难度单词
most_difficult = word_stats.sort_values(by='avg_try', ascending=False).head(n)

# 5. 低难度单词
least_difficult = word_stats.sort_values(by='avg_try', ascending=True).head(m)

# 6. 所有单词按平均尝试次数排序
all_words_sorted = word_stats.sort_values(by='avg_try', ascending=False).reset_index(drop=True)

# 7. 保存结果到txt文件
result_path = os.path.join('difficulty', 'word_difficulty.txt')
with open(result_path, 'w', encoding='utf-8') as f:
    f.write("Wordle Game Difficulty Analysis Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Top {n} words with highest average tries (from difficult to easy):\n")
    f.write("-" * 40 + "\n")
    for index, row in most_difficult.iterrows():
        f.write(f"{row['target']}: {row['avg_try']:.2f} tries, failure rate: {row['failure_rate']:.2f}%\n")
    f.write("\n")
    f.write(f"Top {m} words with lowest average tries (from easy to difficult):\n")
    f.write("-" * 40 + "\n")
    for index, row in least_difficult.iterrows():
        f.write(f"{row['target']}: {row['avg_try']:.2f} tries, failure rate: {row['failure_rate']:.2f}%\n")

# 8. 生成包含排名、单词、平均尝试次数和失败率的CSV文件
print("Generating CSV file...")
csv_path = os.path.join('difficulty', 'word_difficulty_ranking.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    # 9. 写入CSV文件
    csv_writer = csv.writer(csvfile)
    
    # 10. 写入CSV文件头
    csv_writer.writerow(['Ranking', 'Word', 'Average Tries', 'Failure Rate(%)'])
    
    # 11. 写入所有单词数据
    for idx, row in all_words_sorted.iterrows():
        ranking = idx + 1  # 排名从1开始
        csv_writer.writerow([ranking, row['target'], round(row['avg_try'], 2), row['failure_rate']])

print(f"Difficulty analysis results saved to: {result_path}")
print(f"CSV ranking file saved to: {csv_path}")
print("All analysis tasks completed!")