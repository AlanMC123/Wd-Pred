import unittest
import os
import sys
import pandas as pd
import numpy as np
from unittest.mock import patch, mock_open, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import difficulty_analysis
import summary
from difficulty_analysis import FILE_PATH as diff_file_path
from summary import FILE_PATH as summary_file_path

class TestOtherModules(unittest.TestCase):
    """测试其他辅助模块"""
    
    def setUp(self):
        """测试前的准备工作"""
        # 创建测试数据
        self.test_data = {
            'Username': ['user1', 'user1', 'user1', 'user2', 'user2', 'user3'],
            'target': ['apple', 'banana', 'apple', 'banana', 'cherry', 'cherry'],
            'Trial': [3, 4, 7, 2, 5, 6],
            'Game': [1, 2, 3, 4, 5, 6],
            'processed_text': ['text1', 'text2', 'text3', 'text4', 'text5', 'text6']
        }
        self.test_df = pd.DataFrame(self.test_data)
        
        # 创建测试目录
        self.test_dir = 'test_difficulty'
        os.makedirs(self.test_dir, exist_ok=True)
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试目录
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            os.rmdir(self.test_dir)
    
    @patch('difficulty_analysis.pd.read_csv')
    @patch('difficulty_analysis.plt.savefig')
    @patch('difficulty_analysis.plt.close')
    @patch('difficulty_analysis.os.makedirs')
    @patch('difficulty_analysis.os.path.exists')
    @patch('difficulty_analysis.open', new_callable=mock_open)
    def test_difficulty_analysis(self, mock_file, mock_exists, mock_makedirs, mock_close, mock_savefig, mock_read_csv):
        """测试难度分析模块"""
        # 设置mock返回值
        mock_read_csv.return_value = self.test_df
        mock_exists.return_value = False  # 模拟目录不存在，这样os.makedirs会被调用
        mock_makedirs.return_value = None  # 模拟创建目录
        mock_savefig.return_value = None  # 模拟保存图表
        mock_close.return_value = None  # 模拟关闭图表
        
        # 调用difficulty_analysis模块中的函数
        difficulty_analysis.analyze_difficulty()
        
        # 验证功能
        mock_read_csv.assert_called_once_with(diff_file_path)
        mock_savefig.assert_called_once()
        mock_makedirs.assert_called_once_with('difficulty')
        mock_file.assert_called()  # 验证文件写入操作
    
    @patch('summary.pd.read_csv')
    def test_summary(self, mock_read_csv):
        """测试结果汇总模块"""
        # 设置mock返回值
        mock_read_csv.return_value = self.test_df
        
        # 调用summary模块中的函数
        summary.summarize_data()
        
        # 验证功能
        mock_read_csv.assert_called_once_with(summary_file_path)
    
    @patch('summary.pd.read_csv')
    def test_summary_data_info(self, mock_read_csv):
        """测试汇总模块的数据信息功能"""
        # 设置mock返回值
        mock_read_csv.return_value = self.test_df
        
        # 捕获print输出
        import io
        import sys
        from contextlib import redirect_stdout
        
        f = io.StringIO()
        with redirect_stdout(f):
            summary.summarize_data()
        
        output = f.getvalue()
        
        # 验证输出内容
        self.assertIn('数据基本信息：', output)
        self.assertIn('前5行数据：', output)
        self.assertIn('数据统计信息：', output)
        self.assertIn('Game列的唯一值数量：', output)
        self.assertIn('Target列的唯一值数量：', output)

if __name__ == '__main__':
    unittest.main()