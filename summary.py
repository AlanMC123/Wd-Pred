#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

FILE_PATH = 'dataset/cleaned_dataset.csv'

# 加载数据
df = pd.read_csv(FILE_PATH)

# 查看数据基本信息
print('数据基本信息：')
print(df.info())

# 查看前5行数据
print('\\n前5行数据：')
print(df.head())

# 查看数据统计信息
print('\\n数据统计信息：')
print(df.describe())

# 检查Game列的唯一值
print('\\nGame列的唯一值数量：', df['Game'].nunique())
print('Game列的前10个唯一值：', df['Game'].unique()[:10])

# 检查target列的唯一值
print('\\nTarget列的唯一值数量：', df['target'].nunique())
print('Target列的前10个唯一值：', df['target'].unique()[:10])

# 检查processed_text列的结构
print('\\nprocessed_text列的前5行：')
print(df['processed_text'].head())

# 检查processed_text列的长度分布
print('\\nprocessed_text列的长度统计：')
print(df['processed_text'].apply(len).describe())

# 检查Trial列的分布
print('\\nTrial列的分布：')
print(df['Trial'].value_counts())

# 检查Username列的分布
print('\\nUsername列的唯一值数量：', df['Username'].nunique())
print('Username列的前10个唯一值：', df['Username'].unique()[:10])

# 查看一个完整的游戏记录示例
print('\\n一个完整的游戏记录示例：')
# 选择一个有多个尝试的游戏
multi_trial_game = df[df['Trial'] > 1]['Game'].iloc[0]
print(df[df['Game'] == multi_trial_game])