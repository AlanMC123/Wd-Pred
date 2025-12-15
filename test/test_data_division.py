import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_division import MultiThreadedDataProcessor

class TestDataDivision(unittest.TestCase):
    """测试数据分割模块"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.dataset_dir = 'test_dataset'
        self.processor = MultiThreadedDataProcessor(dataset_dir=self.dataset_dir)
        
        # 创建测试数据 - 确保有足够的样本和均匀的成功/失败分布
        self.test_data = {
            'Username': ['user1', 'user1', 'user1', 'user1', 'user1', 'user1', 'user2', 'user2', 'user2', 'user2', 'user2', 'user2'],
            'target': ['apple', 'banana', 'apple', 'banana', 'apple', 'banana', 'apple', 'banana', 'apple', 'banana', 'apple', 'banana'],
            'Trial': [3, 4, 7, 6, 7, 2, 3, 1, 7, 2, 3, 4]
        }
        self.test_df = pd.DataFrame(self.test_data)
        
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试目录
        if os.path.exists(self.dataset_dir):
            for file in os.listdir(self.dataset_dir):
                file_path = os.path.join(self.dataset_dir, file)
                os.remove(file_path)
            os.rmdir(self.dataset_dir)
    
    @patch('data_division.pd.read_csv')
    @patch('data_division.StratifiedShuffleSplit')
    def test_load_and_split_dataset(self, mock_sss, mock_read_csv):
        """测试数据集加载和分割功能"""
        # 设置mock返回值
        mock_read_csv.return_value = self.test_df
        
        # 创建临时的输入文件（因为函数会检查文件是否存在）
        os.makedirs(self.dataset_dir, exist_ok=True)
        with open(self.processor.input_file, 'w') as f:
            f.write('Username,target,Trial\n')
            for _, row in self.test_df.iterrows():
                f.write(f"{row['Username']},{row['target']},{row['Trial']}\n")
        
        # 模拟StratifiedShuffleSplit的行为
        mock_sss_instance = mock_sss.return_value
        
        # 模拟第一次分割 (训练集 vs 临时集)
        def mock_split(X, y):
            if len(X) == len(self.test_df):  # 第一次分割
                # 返回训练集和临时集的索引：12个样本，test_size=0.2，测试集应该有3个样本，训练集有9个样本
                yield [0, 1, 2, 3, 4, 5, 6, 7, 8], [9, 10, 11]
            else:  # 第二次分割 (临时集分割为验证集和测试集)
                # 返回验证集和测试集的索引 - 使用相对于temp_data的索引 [0, 1, 2]
                # temp_data有3个样本，val_size=0.25（相对于训练集+验证集），所以验证集应该有1个样本，测试集有2个样本
                yield [0], [1, 2]
        
        mock_sss_instance.split.side_effect = mock_split
        
        # 调用函数
        train_data, val_data, test_data = self.processor.load_and_split_dataset(test_size=0.2, val_size=0.25)
        
        # 验证结果
        self.assertIsNotNone(train_data)
        self.assertIsNotNone(val_data)
        self.assertIsNotNone(test_data)
        
        # 验证分割结果
        self.assertEqual(len(train_data), 9)  # 9个训练样本
        self.assertEqual(len(val_data), 1)    # 1个验证样本
        self.assertEqual(len(test_data), 2)   # 2个测试样本
        
        # 验证函数被正确调用
        self.assertEqual(mock_sss.call_count, 2)  # 应该调用两次StratifiedShuffleSplit
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(self.processor.train_file))
        self.assertTrue(os.path.exists(self.processor.val_file))
        self.assertTrue(os.path.exists(self.processor.test_file))
    
    def test_calculate_word_difficulty(self):
        """测试单词难度计算功能"""
        # 调用函数
        self.processor.calculate_word_difficulty(self.test_df)
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(self.processor.difficulty_file))
        
        # 验证文件内容
        difficulty_df = pd.read_csv(self.processor.difficulty_file)
        self.assertIn('word', difficulty_df.columns)
        self.assertIn('avg_trial', difficulty_df.columns)
        self.assertIn('failure_rate', difficulty_df.columns)
    
    def test_calculate_player_stats(self):
        """测试玩家统计计算功能"""
        # 调用函数
        self.processor.calculate_player_stats(self.test_df)
        
        # 验证文件是否创建
        self.assertTrue(os.path.exists(self.processor.player_data_file))
        
        # 验证文件内容
        player_df = pd.read_csv(self.processor.player_data_file)
        self.assertIn('Username', player_df.columns)
        self.assertIn('avg_trial', player_df.columns)
        self.assertIn('failure_rate', player_df.columns)
        self.assertIn('total_games', player_df.columns)
    
    def test_file_not_found_error(self):
        """测试文件不存在时的错误处理"""
        # 删除输入文件（如果存在）
        if os.path.exists(self.processor.input_file):
            os.remove(self.processor.input_file)
        
        # 验证异常是否被正确抛出
        with self.assertRaises(FileNotFoundError):
            self.processor.load_and_split_dataset()

if __name__ == '__main__':
    unittest.main()