# Wordle 游戏预测模型

## 项目简介

这是一个用于预测 Wordle 游戏玩家表现的深度学习项目，采用 LSTM 和 Transformer 两种先进的模型架构，通过分析玩家历史游戏数据来预测未来游戏的表现。该项目能够预测玩家完成游戏所需的步数以及成功完成游戏的概率，并提供交互式仪表盘进行可视化展示。

## 目录结构

```
├── difficulty/             # 难度分析结果
│   ├── try_histogram.png   # 尝试次数直方图
│   ├── word_difficulty.txt # 单词难度文本
│   └── word_difficulty_ranking.csv # 单词难度排名
├── models/                 # 训练好的模型
│   ├── lstm/               # LSTM 模型
│   │   ├── lstm_model.keras # LSTM 模型文件
│   │   └── lstm_tokenizer.json # LSTM 分词器
│   └── transformer/        # Transformer 模型
│       └── transformer_model.keras # Transformer 模型文件
├── outputs/                # 模型输出结果
│   ├── LSTM_outcome.txt           # LSTM 整体评估结果
│   ├── LSTM_test_outcome.txt      # LSTM 测试集评估结果
│   ├── LSTM_validation_outcome.txt # LSTM 验证集评估结果
│   ├── Transformer_outcome.txt    # Transformer 整体评估结果
│   ├── Transformer_test_outcome.txt # Transformer 测试集评估结果
│   ├── Transformer_validation_outcome.txt # Transformer 验证集评估结果
│   ├── lstm_output.txt     # LSTM 模型输出
│   └── transformer_output.txt # Transformer 模型输出
├── structure/              # 模型结构可视化
│   ├── lstm_architecture_flat.png # LSTM 架构平面图
│   ├── lstm_structure.txt  # LSTM 结构文本
│   ├── transformer_architecture_flat.png # Transformer 架构平面图
│   └── transformer_structure.txt # Transformer 结构文本
├── visualization/          # 训练和测试结果可视化
│   ├── LSTM_steps_loss_curve.png # LSTM 步数损失曲线
│   ├── LSTM_success_loss_curve.png # LSTM 成功损失曲线
│   ├── LSTM_test_roc_curve.png # LSTM 测试 ROC 曲线
│   ├── LSTM_test_scatter.png # LSTM 测试散点图
│   ├── LSTM_total_loss_curve.png # LSTM 总损失曲线
│   ├── LSTM_validation_roc_curve.png # LSTM 验证 ROC 曲线
│   ├── LSTM_validation_scatter.png # LSTM 验证散点图
│   └── Transformer_test_scatter.png # Transformer 测试散点图
├── dashboard.py            # Streamlit 交互式仪表盘
├── data_division.py        # 数据分割脚本
├── data_seizing.py         # 数据预处理和特征提取脚本
├── difficulty_analysis.py  # 单词难度分析脚本
├── predict.py              # 预测脚本
├── requirements.txt        # 项目依赖
├── structure_visualization.py # 模型结构可视化脚本
├── summary.py              # 结果汇总脚本
├── train_LSTM.py           # LSTM 模型训练脚本
└── train_transformer.py    # Transformer 模型训练脚本
```

## 安装说明

### 环境要求
- Python 3.12.9
- TensorFlow 2.16.0 或更高版本

### 安装步骤

1. 克隆项目仓库：
   ```bash
   git clone <repository-url>
   cd Wd-Pred
   ```

2. 安装项目依赖：
   ```bash
   pip install -r requirements.txt
   ```

3. 确保数据集存在：
   ```
   ├── dataset/
   │   ├── train_data.csv       # 训练集数据
   │   ├── val_data.csv         # 验证集数据
   │   ├── test_data.csv        # 测试集数据
   │   ├── player_data.csv      # 玩家数据
   │   └── difficulty.csv       # 单词难度数据
   ```

## 核心功能

### 1. 数据处理
- **data_division.py**: 将原始数据分割为训练集、验证集和测试集，采用分层抽样确保数据分布均匀
- **data_seizing.py**: 数据预处理和特征提取，包括文本处理、特征工程等
- **difficulty_analysis.py**: 分析单词难度，生成单词难度排名和可视化结果

### 2. 模型训练
- **train_LSTM.py**: 训练 LSTM 模型，使用多输入架构预测玩家表现，包括历史表现序列、单词ID嵌入、用户偏差、单词难度和网格序列特征
- **train_transformer.py**: 训练 Transformer 模型，利用自注意力机制捕获长期依赖关系，采用类似的多输入设计

### 3. 预测功能
- **predict.py**: 使用训练好的模型进行预测，输出预测结果和评估指标
- **dashboard.py**: 交互式仪表盘，支持选择模型类型、玩家ID和单词，实时展示预测结果和可视化

### 4. 可视化与评估
- **structure_visualization.py**: 生成模型结构可视化，便于理解模型架构
- **summary.py**: 汇总模型训练和测试结果，生成评估报告

## 模型架构

### LSTM 模型
- **多输入架构**：
  - 历史表现序列：包含玩家过去5场游戏的表现
  - 单词ID嵌入：将单词转换为向量表示
  - 用户偏差：反映玩家的整体水平
  - 单词难度：反映单词的难易程度
  - 网格序列特征：包含游戏过程中的颜色反馈信息
- **双输出头**：
  - 回归头：预测完成游戏所需的步数
  - 分类头：预测成功完成游戏的概率
- **损失函数**：结合 Focal Loss（分类）和 MSE（回归）
- **正则化**：采用 Dropout 和 L2 正则化防止过拟合

### Transformer 模型
- **自注意力机制**：捕获长期依赖关系，更好地理解玩家历史表现
- **Transformer 编码器块**：包含多头注意力层和前馈神经网络
- **多输入设计**：与 LSTM 模型类似，处理相同的输入特征
- **双输出头**：回归头预测步数，分类头预测成功概率

## 使用方法

### 1. 数据准备

首先运行数据处理脚本，确保数据集已准备好：

```bash
python data_division.py
python data_seizing.py
python difficulty_analysis.py
```

### 2. 模型训练

#### 训练 LSTM 模型：

```bash
python train_LSTM.py
```

训练过程中会生成：
- 训练好的模型文件：`models/lstm/lstm_model.keras`
- 分词器文件：`models/lstm/lstm_tokenizer.json`
- 评估结果：`outputs/LSTM_*.txt`
- 可视化结果：`visualization/LSTM_*.png`

#### 训练 Transformer 模型：

```bash
python train_transformer.py
```

训练过程中会生成：
- 训练好的模型文件：`models/transformer/transformer_model.keras`
- 评估结果：`outputs/Transformer_*.txt`
- 可视化结果：`visualization/Transformer_*.png`

### 3. 运行预测

#### 使用命令行预测：

```bash
python predict.py
```

该脚本支持：
- 选择模型类型（LSTM 或 Transformer）
- 批量预测
- 生成评估报告

#### 使用交互式仪表盘：

```bash
streamlit run dashboard.py
```

在仪表盘上，您可以：
- 选择模型类型（LSTM 或 Transformer）
- 选择玩家 ID
- 选择要预测的单词
- 查看预测结果和可视化
- 比较模型性能

### 4. 生成模型结构可视化

```bash
python structure_visualization.py
```

该脚本会生成：
- 模型架构平面图：`structure/*_architecture_flat.png`
- 模型结构文本：`structure/*_structure.txt`

## 模型评估

### 评估指标

模型评估指标包括：

- **回归任务**：
  - 平均绝对误差 (MAE)
  - 均方根误差 (RMSE)
  - 大误差比例（误差 > 1.5 步）

- **分类任务**：
  - 准确率
  - AUC-ROC
  - 混淆矩阵
  - 精确率、召回率、F1值
  - 失败漏报率

### 最新实验结果

#### Transformer 模型 - 测试集结果

| 指标 | 数值 |
|------|------|
| 平均绝对误差 (MAE) | 0.8208 |
| 均方根误差 (RMSE) | 1.0268 |
| 大误差比例 | 14.04% |
| 准确率 | 87.11% |
| AUC | 0.8792 |
| 失败漏报率 | 30.06% |
| 负类召回率 | 69.94% |
| 负类精确率 | 0.1388 |

#### 混淆矩阵

```
                       Predicted Loss (0)    Predicted Win (1)
------------------------------------------------------------
Actual Loss (0)      |  TN: 121   ( 1.94%) |  FP: 52    ( 0.83%)
Actual Win  (1)      |  FN: 751   (12.05%) |  TP: 5306  (85.17%)
```

## 依赖库

| 库名 | 用途 | 版本要求 |
|------|------|----------|
| tensorflow | 深度学习框架 | >=2.16.0 |
| pandas | 数据处理 | >=2.2.0 |
| numpy | 数值计算 | >=1.26.0 |
| scikit-learn | 机器学习评估工具 | >=1.4.0 |
| matplotlib | 数据可视化 | >=3.8.0 |
| graphviz | 模型结构可视化 | >=0.20.3 |
| wandb | 实验跟踪和可视化 | >=0.16.0 |
| streamlit | 交互式仪表盘 | >=1.30.0 |

## 项目特色

1. **双模型架构**：同时实现了 LSTM 和 Transformer 两种先进的深度学习模型，便于比较不同模型的性能
2. **多输入设计**：结合了多种特征，包括历史表现、单词难度、用户偏差等，提高了预测准确性
3. **双输出头**：同时预测步数和成功概率，满足不同场景的需求
4. **交互式仪表盘**：提供直观的可视化界面，便于用户理解和使用
5. **完整的评估体系**：包含多种评估指标和可视化结果，便于模型优化和比较

## 结论

- Transformer 模型在预测准确率上略优于 LSTM 模型，MAE 为 0.8208，准确率为 87.11%
- 单词难度对玩家表现有显著影响，是重要的预测特征
- 玩家历史表现是预测未来表现的关键因素，LSTM 和 Transformer 都能有效捕获历史依赖关系
- 模型可以有效地预测玩家在 Wordle 游戏中的表现，具有实际应用价值

## 未来改进方向

1. 引入更多玩家特征，如游戏频率、时间分布等，进一步提高预测准确性
2. 尝试更复杂的模型架构，如 GPT 或 BERT 变种，探索预训练模型在该任务上的表现
3. 增加实时数据更新功能，支持模型的持续学习和更新
4. 优化模型训练过程，减少训练时间，提高训练效率
5. 增加更多可视化图表，展示模型学习过程和特征重要性
6. 扩展到其他类似的文字游戏，如 Wordle 的变种或其他语言版本

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！我们鼓励社区贡献，共同改进和扩展这个项目。

## 联系方式

如有问题，请通过以下方式联系：
- 项目仓库：<repository-url>
- 作者：<your-name>

## 致谢

感谢所有为 Wordle 游戏和深度学习社区做出贡献的开发者和研究者！特别感谢 TensorFlow 和 Keras 团队提供的优秀深度学习框架，以及 Streamlit 团队提供的便捷可视化工具。