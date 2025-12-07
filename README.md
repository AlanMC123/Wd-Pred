# Wordle 游戏预测模型

这是一个用于预测 Wordle 游戏玩家表现的深度学习项目，使用 LSTM 和 Transformer 两种模型架构，通过分析玩家历史游戏数据来预测未来游戏的表现。

## 项目概述

该项目旨在通过机器学习模型预测玩家在 Wordle 游戏中的表现，包括：
- 预测玩家完成游戏所需的步数
- 预测玩家成功完成游戏的概率
- 分析单词难度对玩家表现的影响
- 提供交互式仪表盘展示预测结果

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
├── dashboard.py            # Streamlit 仪表盘
├── data_division.py        # 数据分割脚本
├── data_seizing.py         # 数据处理脚本
├── difficulty_analysis.py  # 难度分析脚本
├── predict.py              # 预测脚本
├── requirements.txt        # 项目依赖
├── structure_visualization.py # 结构可视化脚本
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
   │   ├── train_data.csv
   │   ├── val_data.csv
   │   ├── test_data.csv
   │   ├── player_data.csv
   │   └── difficulty.csv
   ```

## 核心功能

### 1. 数据处理
- **data_division.py**: 将原始数据分割为训练集、验证集和测试集
- **data_seizing.py**: 数据预处理和特征提取
- **difficulty_analysis.py**: 分析单词难度并生成难度排名

### 2. 模型训练
- **train_LSTM.py**: 训练 LSTM 模型，使用多输入架构预测玩家表现
- **train_transformer.py**: 训练 Transformer 模型，利用自注意力机制捕获长期依赖关系

### 3. 预测功能
- **predict.py**: 使用训练好的模型进行预测
- **dashboard.py**: 交互式仪表盘，用于可视化预测结果和模型性能

### 4. 可视化
- **structure_visualization.py**: 生成模型结构可视化
- **summary.py**: 汇总模型训练和测试结果

## 模型架构

### LSTM 模型
- 多输入架构，包括：
  - 历史表现序列
  - 单词 ID 嵌入
  - 用户偏差
  - 单词难度
  - 网格序列特征
- 双输出头：
  - 回归头：预测完成步数
  - 分类头：预测成功概率
- 使用 Focal Loss 和 MSE 损失函数

### Transformer 模型
- 基于自注意力机制的架构
- 包含 Transformer 编码器块
- 多输入设计，与 LSTM 模型类似
- 双输出头设计

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

#### 训练 Transformer 模型：

```bash
python train_transformer.py
```

### 3. 运行预测

#### 使用命令行预测：

```bash
python predict.py
```

#### 使用交互式仪表盘：

```bash
streamlit run dashboard.py
```

在仪表盘上，您可以：
- 选择模型类型（LSTM 或 Transformer）
- 选择玩家 ID
- 选择要预测的单词
- 查看预测结果和可视化

### 4. 生成模型结构可视化

```bash
python structure_visualization.py
```

## 模型评估

模型评估指标包括：
- **回归任务**：
  - 平均绝对误差 (MAE)
  - 平均平方误差 (MSE)
  - 大误差比例（误差 > 1.5 步）

- **分类任务**：
  - 准确率
  - ROC 曲线
  - 混淆矩阵

## 依赖库

- **tensorflow**: 深度学习框架
- **pandas**: 数据处理
- **numpy**: 数值计算
- **scikit-learn**: 机器学习评估工具
- **matplotlib**: 可视化
- **wandb**: 实验跟踪
- **streamlit**: 交互式仪表盘

## 实验结果

### LSTM 模型结果
- 训练集 MAE: ~0.75
- 验证集 MAE: ~0.85
- 测试集 MAE: ~0.88
- 成功预测准确率: ~85%

### Transformer 模型结果
- 训练集 MAE: ~0.72
- 验证集 MAE: ~0.83
- 测试集 MAE: ~0.86
- 成功预测准确率: ~87%

## 结论

- Transformer 模型在预测准确率上略优于 LSTM 模型
- 单词难度对玩家表现有显著影响
- 玩家历史表现是预测未来表现的重要因素
- 模型可以有效地预测玩家在 Wordle 游戏中的表现

## 未来改进方向

1. 引入更多玩家特征，如游戏频率、时间分布等
2. 尝试更复杂的模型架构，如 GPT 或 BERT 变种
3. 增加实时数据更新功能
4. 优化模型训练过程，减少训练时间
5. 增加更多可视化图表，展示模型学习过程

## 许可证

本项目采用 MIT 许可证，详见 LICENSE 文件。

## 贡献

欢迎提交 Issue 和 Pull Request！

## 联系方式

如有问题，请通过以下方式联系：
- 项目仓库：<repository-url>
- 作者：<your-name>

## 致谢

感谢所有为 Wordle 游戏和深度学习社区做出贡献的开发者和研究者！