import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_seizing import clean_dataset

class TestDataSeizing(unittest.TestCase):
    """测试数据清洗模块"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.dataset_dir = 'dataset'
        self.input_file = os.path.join(self.dataset_dir, 'wordle_games.csv')
        self.output_file = os.path.join(self.dataset_dir, 'cleaned_dataset.csv')
        
        # 创建测试数据
        self.test_data = {
            'Username': ['user1', 'user1', 'user1', 'user1', 'user1', 'user2', 'user2', 'user3'],
            'Trial': [3, 4, 5, 6, 7, 2, 3, 1],
            'target': ['apple', 'banana', 'cherry', 'date', 'elder', 'fig', 'grape', 'honey']
        }
        self.test_df = pd.DataFrame(self.test_data)
        
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试生成的文件
        for file in [self.output_file]:
            if os.path.exists(file):
                os.remove(file)
        
    @patch('data_seizing.pd.read_csv')
    @patch('data_seizing.pd.DataFrame.to_csv')
    @patch('data_seizing.random.sample')
    def test_clean_dataset(self, mock_sample, mock_to_csv, mock_read_csv):
        """测试数据集清洗功能"""
        # 设置mock返回值
        mock_read_csv.return_value = self.test_df
        mock_sample.return_value = ['user1']  # 只选择user1进行测试
        
        # 调用函数
        result = clean_dataset(0.5)  # 50%的用户
        
        # 验证结果
        self.assertEqual(result, self.output_file)
        mock_read_csv.assert_called_once_with(self.input_file)
        
        # 验证随机采样 - 实际只有user1参与了5次以上游戏，所以只从符合条件的用户中采样
        eligible_users = ['user1']
        mock_sample.assert_called_once_with(eligible_users, 1)  # 1个符合条件的用户的50%是0.5，取整为1
        mock_to_csv.assert_called_once()
    
    @patch('data_seizing.pd.read_csv')
    def test_file_not_found(self, mock_read_csv):
        """测试文件不存在的情况"""
        # 设置mock抛出异常
        mock_read_csv.side_effect = FileNotFoundError("找不到文件")
        
        # 验证异常是否被正确处理
        with self.assertRaises(FileNotFoundError):
            clean_dataset(0.5)
        
        mock_read_csv.assert_called_once_with(self.input_file)

if __name__ == '__main__':
    unittest.main()