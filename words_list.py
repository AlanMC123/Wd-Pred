import csv

# 文件路径
FILE_PATH = 'F:\Codes\wordle_games.csv'
OUTPUT_FILE = 'words.txt'

def process_csv_file():
    """读取CSV文件，统计target列中的单词"""
    # 使用集合存储唯一的单词
    unique_words = set()
    
    try:
        # 打开CSV文件进行读取
        with open(FILE_PATH, 'r', encoding='utf-8') as csv_file:
            # 创建CSV读取器
            csv_reader = csv.DictReader(csv_file)
            
            # 检查是否存在target列
            if 'target' not in csv_reader.fieldnames:
                print(f"错误: CSV文件中不存在'target'列")
                return False
            
            # 遍历CSV文件的每一行
            for row in csv_reader:
                # 获取target列的值并添加到集合中
                target_word = row['target'].strip()
                if target_word:
                    unique_words.add(target_word)
        
        # 将唯一单词写入输出文件
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as output_file:
            for word in sorted(unique_words):
                output_file.write(word + '\n')
        
        print(f"成功: 已统计{len(unique_words)}个唯一单词，并写入{OUTPUT_FILE}")
        return True
        
    except FileNotFoundError:
        print(f"错误: 找不到文件{FILE_PATH}")
        return False
    except Exception as e:
        print(f"错误: 处理CSV文件时发生异常: {e}")
        return False

if __name__ == "__main__":
    process_csv_file()