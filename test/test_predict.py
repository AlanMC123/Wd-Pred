import unittest
import os
import sys
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from unittest.mock import patch, mock_open, MagicMock

# 将项目根目录添加到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from predict import (
    set_seed, TransformerBlock, focal_loss, safe_read_csv, load_tokenizer,
    encode_guess_sequence, attach_features, build_history, create_samples_lstm,
    create_samples_transformer, calculate_metrics
)

class TestPredict(unittest.TestCase):
    """测试预测模块"""
    
    def setUp(self):
        """测试前的准备工作"""
        self.test_dir = 'test_predict_outputs'
        os.makedirs(self.test_dir, exist_ok=True)
        
        # 创建测试数据
        self.test_df = pd.DataFrame({
            'Username': ['user1', 'user1', 'user1', 'user1', 'user1', 'user1'],
            'target': ['apple', 'banana', 'cherry', 'date', 'elder', 'fig'],
            'Trial': [3, 4, 5, 6, 7, 2],
            'Game': [1, 2, 3, 4, 5, 6]
        })
        
        # 创建测试tokenizer
        self.test_tokenizer = {
            'apple': 1,
            'banana': 2,
            'cherry': 3,
            'date': 4,
            'elder': 5,
            'fig': 6,
            '<OOV>': 0
        }
        
        # 用户和难度映射
        self.user_map = {'user1': 3.5}
        self.diff_map = {'apple': 4.0, 'banana': 3.0, 'cherry': 5.0, 'date': 3.5, 'elder': 4.5, 'fig': 2.5}
    
    def tearDown(self):
        """测试后的清理工作"""
        # 删除测试目录
        if os.path.exists(self.test_dir):
            for file in os.listdir(self.test_dir):
                file_path = os.path.join(self.test_dir, file)
                os.remove(file_path)
            os.rmdir(self.test_dir)
    
    def test_set_seed(self):
        """测试随机种子设置功能"""
        # 设置不同种子并生成随机数
        set_seed(42)
        val1 = np.random.rand()
        
        set_seed(42)
        val2 = np.random.rand()
        
        set_seed(123)
        val3 = np.random.rand()
        
        # 验证相同种子生成相同结果
        self.assertEqual(val1, val2)
        # 验证不同种子生成不同结果
        self.assertNotEqual(val1, val3)
    
    def test_transformer_block(self):
        """测试TransformerBlock层"""
        # 创建TransformerBlock实例
        embed_dim = 64
        num_heads = 8
        ff_dim = 128
        transformer_block = TransformerBlock(embed_dim=embed_dim, num_heads=num_heads, ff_dim=ff_dim)
        
        # 测试前向传播
        input_data = tf.random.normal(shape=(32, 5, embed_dim))
        output = transformer_block(input_data)
        
        # 验证输出形状
        self.assertEqual(output.shape, (32, 5, embed_dim))
    
    def test_focal_loss(self):
        """测试focal_loss函数"""
        # 创建focal_loss实例
        loss_fn = focal_loss(gamma=2.0, alpha=0.25)
        
        # 测试损失计算
        y_true = tf.constant([1.0, 0.0, 1.0, 0.0])
        y_pred = tf.constant([0.9, 0.1, 0.8, 0.2])
        loss = loss_fn(y_true, y_pred)
        
        # 验证损失值是一个标量
        self.assertEqual(tf.rank(loss).numpy(), 0)
    
    def test_safe_read_csv(self):
        """测试安全读取CSV文件功能"""
        # 测试文件不存在的情况
        df_missing = safe_read_csv('non_existent_file.csv')
        self.assertTrue(df_missing.empty)
        
        # 测试文件存在的情况
        test_csv = os.path.join(self.test_dir, 'test.csv')
        self.test_df.to_csv(test_csv, index=False)
        
        df_exist = safe_read_csv(test_csv)
        self.assertEqual(len(df_exist), len(self.test_df))
    
    @patch('predict.os.path.exists')
    @patch('predict.open', new_callable=mock_open, read_data=json.dumps({'apple': 1, 'banana': 2, '<OOV>': 0}))
    def test_load_tokenizer(self, mock_file, mock_exists):
        """测试加载tokenizer功能"""
        # 设置mock
        mock_exists.return_value = True
        
        # 调用函数
        tokenizer = load_tokenizer('test_tokenizer.json')
        
        # 验证结果
        self.assertIsNotNone(tokenizer)
        self.assertEqual(tokenizer.word_index['apple'], 1)
        mock_exists.assert_called_once_with('test_tokenizer.json')
    
    def test_encode_guess_sequence(self):
        """测试猜测序列编码功能"""
        # 测试有效的猜测序列
        grid_cell = "['🟩🟩🟩🟩🟩', '🟩🟨⬜⬜⬜', '🟩🟨🟨⬜⬜']"
        sequence = encode_guess_sequence(grid_cell)
        
        # 验证输出形状
        self.assertEqual(sequence.shape, (6, 8))
        
        # 测试空值情况
        sequence_null = encode_guess_sequence(None)
        self.assertEqual(sequence_null.shape, (6, 8))
        
        # 测试无效格式
        sequence_invalid = encode_guess_sequence('invalid_format')
        self.assertEqual(sequence_invalid.shape, (6, 8))
    
    @patch('predict.Tokenizer')
    def test_attach_features(self, mock_tokenizer_class):
        """测试特征附加功能"""
        # 设置mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.texts_to_sequences.return_value = [[1], [2], [3], [4], [5], [6]]
        mock_tokenizer_class.return_value = mock_tokenizer
        
        # 调用函数
        result_df = attach_features(self.test_df, mock_tokenizer, self.user_map, self.diff_map)
        
        # 验证结果
        self.assertIn('word_id', result_df.columns)
        self.assertIn('word_difficulty', result_df.columns)
        self.assertIn('user_bias', result_df.columns)
        self.assertIn('grid_seq_processed', result_df.columns)
    
    def test_build_history(self):
        """测试构建历史记录功能"""
        # 添加build_history函数所需的所有列
        self.test_df['word_id'] = [1, 2, 3, 4, 5, 6]  # 添加word_id列
        self.test_df['user_bias'] = [3.5, 3.5, 3.5, 3.5, 3.5, 3.5]  # 添加user_bias列
        self.test_df['word_difficulty'] = [4.0, 3.0, 5.0, 3.5, 4.5, 2.5]  # 添加word_difficulty列
        self.test_df['grid_seq_processed'] = [np.zeros((6, 8)) for _ in range(len(self.test_df))]  # 添加grid_seq_processed列
        
        # 调用函数
        history = build_history(self.test_df)
        
        # 验证结果
        self.assertIsInstance(history, dict)
        self.assertIn('user1', history)
        self.assertEqual(len(history['user1']), len(self.test_df))
    
    def test_calculate_metrics(self):
        """测试指标计算功能"""
        # 创建测试数据
        y_true_steps = np.array([3.0, 4.0, 5.0, 6.0, 7.0, 2.0])
        y_true_succ = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 1.0])
        pred_steps = np.array([3.2, 4.1, 4.8, 6.3, 6.7, 2.2])
        pred_prob = np.array([0.8, 0.9, 0.7, 0.85, 0.3, 0.95])
        threshold = 0.69
        
        # 调用函数
        metrics = calculate_metrics(y_true_steps, y_true_succ, pred_steps, pred_prob, threshold)
        
        # 验证结果
        self.assertIn('MAE', metrics)
        self.assertIn('RMSE', metrics)
        self.assertIn('ACC', metrics)
        self.assertIn('AUC', metrics)
        self.assertIn('CM', metrics)
        
        # 验证指标值的合理性
        self.assertGreater(metrics['MAE'], 0)
        self.assertGreater(metrics['RMSE'], 0)
        self.assertLessEqual(metrics['ACC'], 1.0)
        self.assertLessEqual(metrics['AUC'], 1.0)

if __name__ == '__main__':
    unittest.main()