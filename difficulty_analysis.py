import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
import csv

FILE_PATH = 'dataset/cleaned_dataset.csv'

# Set font to Arial for better English display
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

# Create difficulty folder
if not os.path.exists('difficulty'):
    os.makedirs('difficulty')

# Read CSV file
print("Reading wordle_games.csv file...")
df = pd.read_csv(FILE_PATH)

# 1. Output Try distribution histogram
print("Generating try distribution histogram...")
plt.figure(figsize=(12, 6))
plt.hist(df['Trial'], bins=range(1, df['Trial'].max() + 2), edgecolor='black', align='left')
plt.title('Wordle Try Distribution')
plt.xlabel('Number of Tries')
plt.ylabel('Number of Games (10,000)')
plt.xticks(range(1, df['Trial'].max() + 1))
# Set y-axis unit to 10,000
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: '{:.1f}'.format(x/10000)))
plt.grid(axis='y', alpha=0.75)
plt.tight_layout()

# Save histogram
histogram_path = os.path.join('difficulty', 'try_histogram.png')
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Try distribution histogram saved to: {histogram_path}")

# 2. Calculate average Tries and failure rate for each target word
print("Calculating average Tries and failure rate for each target word...")

# Calculate average Tries
target_avg_try = df.groupby('target')['Trial'].mean().reset_index()
target_avg_try.columns = ['target', 'avg_try']

# Calculate total games
total_games = df.groupby('target').size().reset_index(name='total_games')

# Calculate failed games (assuming Trial = 7 indicates failure)
failed_games = df[df['Trial'] == 7].groupby('target').size().reset_index(name='failed_games')

# Merge data
word_stats = pd.merge(target_avg_try, total_games, on='target')
word_stats = pd.merge(word_stats, failed_games, on='target', how='left')

# Fill missing values (words with no failed records)
word_stats['failed_games'] = word_stats['failed_games'].fillna(0)

# Calculate failure rate
word_stats['failure_rate'] = (word_stats['failed_games'] / word_stats['total_games'] * 100).round(2)

# 3. Set values for n and m
n = 10  # Top n words with highest average Tries
m = 10  # Top m words with lowest average Tries

# 4. Find top n words with highest average Tries
most_difficult = word_stats.sort_values(by='avg_try', ascending=False).head(n)

# 5. Find top m words with lowest average Tries
least_difficult = word_stats.sort_values(by='avg_try', ascending=True).head(m)

# 6. Prepare data for all words, sorted by average Tries in descending order
all_words_sorted = word_stats.sort_values(by='avg_try', ascending=False).reset_index(drop=True)

# 6. Save results to txt file
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

# 7. Generate CSV file with ranking, word, average Tries, and failure rate
print("Generating CSV file...")
csv_path = os.path.join('difficulty', 'word_difficulty_ranking.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
    # Create CSV writer
    csv_writer = csv.writer(csvfile)
    
    # Write header
    csv_writer.writerow(['Ranking', 'Word', 'Average Tries', 'Failure Rate(%)'])
    
    # Write data, ranking starts from 1
    for idx, row in all_words_sorted.iterrows():
        ranking = idx + 1  # Ranking starts from 1
        csv_writer.writerow([ranking, row['target'], round(row['avg_try'], 2), row['failure_rate']])

print(f"Difficulty analysis results saved to: {result_path}")
print(f"CSV ranking file saved to: {csv_path}")
print("All analysis tasks completed!")