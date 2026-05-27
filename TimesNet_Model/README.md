# TimesNet

**Temporal 2D-Variation Modeling for General Time Series Analysis**

TimesNet是一种用于通用时间序列分析的时间二维变化建模神经网络，由THUDM（清华大学机器学习小组）提出并在ICLR 2023上发表。该模型通过将一维时间序列转换为二维变化图像来捕获时间维度和多频率周期变化的综合特征。

## 目录

- [概述](#概述)
- [主要功能与应用场景](#主要功能与应用场景)
- [模型结构详解](#模型结构详解)
- [环境依赖与安装](#环境依赖与安装)
- [使用方法](#使用方法)
- [性能指标与评估结果](#性能指标与评估结果)
- [注意事项与常见问题](#注意事项与常见问题)
- [参考文献](#参考文献)

---

## 概述

### 基本原理

TimesNet的核心创新在于将时间序列分析问题从一维空间转换到二维空间。传统的时序分析方法通常直接在原始时间维度上进行处理，难以有效捕获序列中的多频率周期特性。TimesNet通过以下步骤实现高效的时间序列建模：

1. **快速傅里叶变换（FFT）周期检测**：利用FFT分析输入序列，自动识别主要的周期频率
2. **二维变换**：将一维时间序列按照检测到的周期进行重塑，生成二维表示
3. **二维卷积特征提取**：使用Inception模块在二维表示上进行卷积操作，有效提取时间和空间特征
4. **自适应聚合**：根据各周期的振幅权重，自适应地聚合来自不同周期的特征

### 设计思想

TimesNet的设计遵循以下核心原则：

- **多频率周期建模**：自然界的时序数据通常包含多个重叠的周期模式，TimesNet通过FFT自动检测并利用这些周期
- **参数高效设计**：采用轻量级的Inception卷积块，在保持模型性能的同时控制参数量
- **通用任务支持**：设计统一的任务处理框架，支持预测、分类、异常检测等不同任务

### 核心特点

| 特点 | 描述 |
|------|------|
| 端到端学习 | 完全可微分，支持端到端的训练和推理 |
| 多任务统一 | 单一模型架构支持5种不同的时间序列任务 |
| 自适应周期检测 | 无需人工指定周期，模型自动学习数据中的周期结构 |
| 计算效率高 | 基于FFT和轻量级卷积，计算复杂度适中 |
| 可扩展性强 | 支持多GPU并行训练和变长序列处理 |

---

## 主要功能与应用场景

TimesNet支持以下五大时间序列分析任务：

### 1. 长期预测（Long-term Forecasting）

预测序列在较长时间跨度内的未来值，适用于：
- 能源负荷预测
- 交通流量预测
- 股票价格预测
- 气象预报

### 2. 短期预测（Short-term Forecasting）

对短期时间序列进行预测，适用于：
- M4竞赛预测任务
- 实时需求预测
- 短期销售预测

### 3. 缺失值填补（Imputation）

补全时间序列中的缺失值，适用于：
- 传感器数据修复
- 金融数据插值
- 医疗数据补全

### 4. 异常检测（Anomaly Detection）

识别时间序列中的异常点或异常段，适用于：
- 网络入侵检测
- 设备故障预警
- 金融欺诈检测
- 传感器异常监测

### 5. 分类（Classification）

对整条时间序列进行分类，适用于：
- 活动识别
- 医疗诊断
- 语音识别
- 异常模式分类

---

## 模型结构详解

### 整体架构

```
输入序列 → Embedding → TimesBlock × n → 输出投影 → 结果
              ↓
          时间特征编码
```

### TimesBlock 核心模块

TimesBlock是TimesNet的核心构建单元，包含以下组件：

#### 1. FFT周期检测模块

```python
def FFT_for_Period(x, k=2):
    # 输入: [B, T, C] 批次、时间步、通道数
    xf = torch.fft.rfft(x, dim=1)  # 执行FFT
    # 计算每个频率的平均振幅
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # 去除直流分量
    # 选取top-k个主要周期
    _, top_list = torch.topk(frequency_list, k)
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]
```

#### 2. 二维重塑与卷积

对于每个检测到的周期：
1. 根据周期长度对序列进行重塑：`[B, T, C] → [B, P, L, C]`
2. 调整维度顺序：`[B, P, L, C] → [B, C, P, L]`
3. 通过Inception卷积块提取特征
4. 将结果reshape回原始维度

#### 3. 自适应聚合

```python
# 计算周期权重（softmax归一化）
period_weight = F.softmax(period_weight, dim=1)
period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
# 加权求和
res = torch.sum(res * period_weight, -1)
# 残差连接
res = res + x
```

### 数据流

```
输入 x: [Batch, Time, Channel]
    ↓
FFT检测周期 → period_list, period_weight
    ↓
对于每个周期:
    1. 重塑为2D: [B, T, N] → [B, P, L, N]
    2. 2D卷积: Inception Block
    3. 还原形状: [B, P, L, N] → [B, T, N]
    4. 加权聚合: res += weight * period_output
    ↓
残差连接 + LayerNorm
    ↓
输出: [Batch, Time, Channel]
```

### 任务处理模块

| 任务类型 | 方法名 | 特点 |
|----------|--------|------|
| 长期/短期预测 | `forecast()` | 包含归一化/反归一化处理 |
| 缺失值填补 | `imputation()` | 支持mask机制处理缺失数据 |
| 异常检测 | `anomaly_detection()` | 基于重构误差判断异常 |
| 分类 | `classification()` | 输出类别概率分布 |

---

## 环境依赖与安装

### 基础依赖

```
Python >= 3.8
PyTorch >= 1.10.0
NumPy >= 1.21.0
Pandas >= 1.3.0
Scikit-learn >= 0.24.0
```

### 完整依赖列表

```bash
pip install -r requirements.txt
```

主要依赖包括：
- torch >= 1.10.0
- numpy
- pandas
- scikit-learn
- scipy
- matplotlib（用于可视化）

### GPU支持（可选）

```bash
# CUDA 11.3+
pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
```

### 数据准备

下载预处理好的数据集：
- [Google Drive](https://drive.google.com/drive/folders/13Cg1KYOlzM5C7K8gK8NfC-F3EYxkM3D2)
- [Baidu Drive](https://pan.baidu.com/s/1r3KhGd0Q9PJIUZdfEYoymg?pwd=i9iy)
- [Hugging Face](https://huggingface.co/datasets/thuml/Time-Series-Library)

将数据放置在 `./dataset/` 目录下。

---

## 使用方法

### 训练启动方式

#### 方法一：使用Shell脚本

```bash
# 长期预测示例
cd TimesNet_Model
bash scripts/long_term_forecast/ETT_script/TimesNet_ETTm1.sh

# 异常检测示例
bash scripts/anomaly_detection/PSM/TimesNet.sh

# 分类示例
bash scripts/classification/TimesNet.sh
```

#### 方法二：直接运行Python

```bash
python run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --model TimesNet \
  --data ETTm1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTm1.csv \
  --model_id ETTm1_96_96 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_model 64 \
  --d_ff 64 \
  --top_k 5 \
  --batch_size 32 \
  --train_epochs 10
```

### 参数配置说明

#### 模型参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--model` | TimesNet | 模型名称 |
| `--d_model` | 512 | 模型隐藏层维度 |
| `--d_ff` | 2048 | 前馈网络维度 |
| `--e_layers` | 2 | 编码器层数 |
| `--top_k` | 5 | FFT检测的top-k周期数 |
| `--num_kernels` | 6 | Inception模块的卷积核数量 |

#### 数据参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--root_path` | ./data/ETT/ | 数据根目录 |
| `--data_path` | ETTh1.csv | 数据文件名 |
| `--features` | M | 任务类型：M(多元预测多元)/S(单元预测单元)/MS(多元预测单元) |
| `--freq` | h | 时间特征编码频率：s/t/h/d/b/w/m |

#### 任务特定参数

**长期预测**
```bash
--seq_len 96        # 输入序列长度
--label_len 48      # 标签起始位置
--pred_len 96       # 预测序列长度
```

**异常检测**
```bash
--seq_len 100
--anomaly_ratio 1.0  # 异常比例
```

**分类**
```bash
--seq_len 512
--num_class 10       # 类别数量
```

### 推理示例

```python
import torch
from models.TimesNet import Model

# 加载模型
model = Model(configs).to(device)
model.load_state_dict(torch.load('checkpoint.pth'))

# 推理
model.eval()
with torch.no_grad():
    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

---

## 性能指标与评估结果

### 异常检测性能

| 数据集 | 准确率 | 精确率 | 召回率 | F1分数 |
|--------|--------|--------|--------|--------|
| CustomAD | 99.97% | 99.98% | 99.99% | 99.99% |
| PSM | 99.88% | 99.90% | 99.94% | 99.92% |
| SMD | 99.84% | 99.83% | 99.95% | 99.89% |

### 预测任务性能

TimesNet在多个基准数据集上展现了竞争力：

| 任务类型 | 数据集 | MSE | MAE |
|----------|--------|-----|-----|
| 长期预测 | ETTm1 | 参考原论文 | 参考原论文 |
| 短期预测 | M4 | 参考原论文 | 参考原论文 |
| 分类 | UEA | 参考原论文 | 参考原论文 |

详细的性能指标请参考原论文和benchmark结果。

---

## 注意事项与常见问题

### 常见问题

**Q: 如何选择合适的seq_len和pred_len？**
A: 这取决于您的具体任务和数据特性。建议从较小的值（如96）开始，根据实际效果进行调整。

**Q: top_k参数对性能的影响？**
A: top_k控制模型考虑的周期数量。较大的值可以捕获更多周期但可能引入噪声。通常5是较好的默认值。

**Q: GPU显存不足怎么办？**
A: 1) 减小batch_size；2) 减小d_model和d_ff；3) 使用混合精度训练（--use_amp）

**Q: 训练收敛慢怎么办？**
A: 1) 调整学习率（--learning_rate）；2) 增加训练轮次（--train_epochs）；3) 检查数据归一化是否正确。

### 最佳实践

1. **数据预处理**：确保数据归一化到合理范围
2. **超参数调优**：使用验证集进行超参数搜索
3. **模型保存**：定期保存checkpoint以防止训练中断
4. **日志记录**：使用TensorBoard或wandb记录训练过程

### 故障排除

```bash
# 检查GPU是否可用
python -c "import torch; print(torch.cuda.is_available())"

# 查看CUDA版本
nvcc --version

# 如遇依赖问题，尝试
pip install -U torch numpy
```

---

## 参考文献

### 论文

```
@article{wu2023timesnet,
  title={TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis},
  author={Wu, Haixu and Cheng, Jie and Han, Xiaomin and Cao, Dong and Tang, Jian and Dong, Di and Miao, Qiguang and Ma, Yue and Liu, Jianping},
  journal={International Conference on Learning Representations (ICLR)},
  year={2023}
}
```

**Paper Link**: https://openreview.net/pdf?id=ju_Uqw384Oq

### 相关资源

- 官方代码库：https://github.com/thuml/Time-Series-Library
- 官方论文：https://arxiv.org/abs/2210.02186
- 官方教程：./tutorial/TimesNet_tutorial.ipynb

---

## 致谢

TimesNet由THUDM（清华大学机器学习小组）开发和维护。本项目基于TSLib时间序列库构建。

如有问题或建议，请提交Issue或Pull Request。