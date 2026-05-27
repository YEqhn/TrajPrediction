# TCN关节轨迹预测模型

本项目实现了一个基于时间卷积网络（Temporal Convolutional Network, TCN）的关节轨迹预测模型，利用惯性测量单元（IMU）传感器数据进行关节角度预测。该模型在生物力学分析、人体运动重建和人机交互等领域具有重要应用价值。

## 目录

- [项目概述](#项目概述)
- [代码文件功能说明](#代码文件功能说明)
- [模型输入输出规范](#模型输入输出规范)
- [训练日志说明](#训练日志说明)
- [补充说明](#补充说明)

---

## 项目概述

### 研究背景

传统的关节角度测量主要依赖昂贵的光学运动捕捉系统或接触式角度传感器，这些方法在室外环境、自由活动场景和长时间监测中面临诸多限制。IMU传感器以其体积小、重量轻、成本低和佩戴方便的特点，成为替代方案的理想选择。然而，IMU原始数据包含噪声且存在漂移问题，直接用于关节角度估计存在精度不足的挑战。

### 应用场景

- **康复医学**：实时监测患者关节活动范围，评估康复训练效果
- **运动分析**：分析运动员动作技术，辅助优化训练方案
- **人机交互**：为虚拟现实和机器人控制提供自然的人体运动估计
- **生物力学研究**：在实验室外环境中进行长期人体运动数据采集

### 核心价值

本项目采用TCN架构处理时序IMU数据，能够有效捕捉传感器数据中的时间依赖关系，实现高精度的关节角度预测。模型采用滑动窗口机制提取时域特征，支持自定义预测时域以满足不同应用需求。

---

## 代码文件功能说明

### prediction_hip_test.py（主训练脚本）

主程序入口，负责模型训练的全流程管理。脚本首先解析命令行参数以配置训练超参数，包括批次大小（默认32）、学习率（默认1e-3）、训练轮次（默认50）和网络结构参数（隐藏层数6层、每层64个通道、核大小5）。核心流程包括：数据加载与预处理、模型初始化、训练循环执行和模型权重保存。训练过程中每50个批次输出一次日志，包含当前损失值、MAE和RMSE等评估指标，每10个epoch自动将学习率减半以促进收敛。

### model.py（模型定义）

实现TCN回归器模型架构，继承PyTorch的nn.Module基类。TCNRegressor类包含时序卷积网络层（TemporalConvNet）和线性输出层，接收IMU特征序列并输出关节角度预测。模型使用MSE损失函数进行训练，通过均方根误差（RMSE）评估预测精度。此外提供了MultiStepTCNRegressor类支持多步预测，可预测未来多个时间点的关节角度。forward方法中，时间卷积网络的输出取最后一个时间步的特征，通过线性层映射到目标关节维度。

### utils.py（数据处理工具）

提供数据加载、预处理和评估相关的工具函数。load_dataset_data函数从指定路径读取IMU和Goniometer（角度仪）配对数据，执行Z-score标准化后返回训练集和测试集。create_sliding_window_sequences函数实现滑动窗口机制，根据配置的窗口大小（默认1秒）和预测时域（默认100毫秒）生成训练样本。data_generator函数实现批量数据加载器，支持随机打乱和数据批量化。denormalize_*系列函数负责将归一化空间的预测误差转换回原始角度单位（度），以便直观评估模型性能。

### tcn.py（TCN基础架构）

实现Temporal Convolutional Network的核心组件。TemporalBlock类包含两个堆叠的一维卷积层，每层后接ReLU激活和Dropout正则化，采用残差连接结构以缓解梯度消失问题。Chomp1d类裁剪多余的填充以确保因果卷积特性。TemporalConvNet类堆叠多个TemporalBlock，每层的膨胀系数（dilation）以2的幂次递增，使网络能够捕获更长范围的时序依赖。权重归一化（weight_norm）应用于卷积层以稳定训练过程。该架构相比传统RNN具有并行计算优势，训练效率更高。

---

## 模型输入输出规范

### 输入规范

**数据文件路径**：`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/imu/`

**数据文件名格式**：`ramp_{ramp_num}_{side}_{trial}_{subtrial}.csv`
- ramp_num：坡道编号（1, 2等）
- side：左右侧（l表示左腿left，r表示右腿right）
- trial：试次编号（01, 02, 03等）
- subtrial：子试次编号（01, 02等）

**示例文件名**：`ramp_1_l_01_01.csv`、`ramp_2_r_03_02.csv`

**数据文件内部结构**：

- 第一列（索引0）：时间戳（timestamp）
- 第二至第二十五列（索引1-24）：24个IMU特征通道，来自4个传感器节点：
  - 索引1-6：足部IMU（foot_Accel_X/Y/Z + foot_Gyro_X/Y/Z）
  - 索引7-12：小腿IMU（shank_Accel_X/Y/Z + shank_Gyro_X/Y/Z）
  - 索引13-18：大腿IMU（thigh_Accel_X/Y/Z + thigh_Gyro_X/Y/Z）
  - 索引19-24：躯干IMU（trunk_Accel_X/Y/Z + trunk_Gyro_X/Y/Z）
- 列标题行：包含各列的特征名称（Header, foot_Accel_X, ...）

**数据维度信息**：

- 单个CSV文件行数：通常10000-16000行（取决于记录时长）
- 输入张量形状：`(batch_size, 24, window_samples)`
- batch_size：每批次处理的样本数量，默认32
- 24：IMU特征通道数（4个IMU × 6通道 = 24个传感器特征）
- window_samples：滑动窗口内的采样点数量

**参数含义**：

- `window_samples = int(window_seconds × sample_rate)`，默认window_seconds=1.0、sample_rate=200Hz时，window_samples=200
- 输入数据经过Z-score标准化处理：(data - mean) / std

### 输出规范

**数据文件路径**：`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/gon/`

**数据文件名格式**：与IMU数据文件一一对应，使用相同的命名规则（如`ramp_1_l_01_01.csv`）

**数据文件内部结构**：

- 第一列（索引0）：时间戳（timestamp）
- 后续列：包含多个关节角度特征，其中模型仅使用包含"sagittal"关键字的特征列
- 列标题行：包含各列的特征名称

**输出特征选择**：

模型仅提取包含"sagittal"关键字的关节角度列作为训练目标。典型配置下包含3个sagittal关节角度特征，分别对应髋关节和膝关节的矢状面运动（如髋关节屈伸、膝关节屈伸等）。

**数据维度信息**：

- 输出张量形状：`(batch_size, 3)`
- 3：sagittal关节角度通道数（固定值）
- 输出值单位：度（degrees），经过反归一化处理后

**参数含义**：

- output_size：模型预测的关节角度通道数，默认值为3
- 输出值范围：反归一化后为实际角度值，典型范围-90°至90°

### 训练样本生成机制

**滑动窗口机制**：

1. 从原始IMU和GON数据中，按时间顺序滑动窗口
2. 输入窗口：连续`window_samples`个IMU采样点
3. 预测目标：`window_samples + predict_samples`位置处的关节角度
4. predict_samples = int(predict_ms × sample_rate / 1000)，默认predict_ms=100ms、sample_rate=200Hz时，predict_samples=20

**训练样本示例**（默认配置）：

- 输入：200个连续的IMU采样点（1秒数据）
- 输出：100毫秒后的3个sagittal关节角度

---

## 训练日志说明

### 日志类型与记录频率

**训练批次日志**（每50个批次打印一次）：

- 训练轮次与进度：`Train Epoch: {epoch} [{batch_idx} / {total_batches} ({percentage}%)]`
- 归一化损失：`Loss: {value}` （归一化空间的MSE值）
- 反归一化误差指标：`MSE: {value}°²`、`MAE: {value}°`、`RMSE: {value}°`
- 训练耗时：`Time: {elapsed}s`

**验证日志**（每个epoch结束后）：

- 验证集指标：`Validation | MSE: {value}°² | MAE: {value}° | RMSE: {value}°`
- 最优模型保存通知：当验证损失低于历史最优时输出

**Epoch汇总日志**（每个epoch结束）：

- Epoch进度：`Epoch {current}/{total} completed in {time}s`
- 训练损失和验证损失：`Train Loss: {value} | Val Loss: {value}`

**训练完成日志**：

- 总训练时长：`Total training time: {value}s`
- 最优验证指标：`Best validation MSE: {value}°² (RMSE: {value}°)`
- 模型和归一化参数保存路径

### 数据格式与阈值范围

**归一化空间指标**：

- Loss值范围通常在0.001至1.0之间，训练后期应降至0.01以下

**反归一化空间指标（单位：度）**：

- MSE：均方误差，典型范围5-50°²，训练目标应低于20°²
- MAE：平均绝对误差，典型范围2-15°，训练目标应低于5°
- RMSE：均方根误差，典型范围2-10°，训练目标应低于5°

### 关键指标含义

**MSE（均方误差）**：预测误差平方的均值，对大误差更加敏感，用于评估模型整体精度。

**MAE（平均绝对误差）**：预测误差绝对值的均值，对异常值鲁棒，提供误差幅度的直观度量。

**RMSE（均方根误差）**：MSE的平方根，与原始数据单位一致，便于解释，是评估关节角度预测精度的首选指标。

---

## 数据集说明

### 训练数据集

**数据存储路径**：`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/`

**IMU传感器数据目录**：`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/imu/`

**Goniometer参考数据目录**：`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/gon/`

**数据文件配对规则**：

IMU目录和GON目录中的CSV文件通过文件名一一匹配。只有两个目录中都存在的文件才会被加载用于训练。系统自动匹配两个目录中的文件，生成配对数据集。

**典型数据文件列表**：

| 文件名 | 含义 | 数据类型 |
|--------|------|----------|
| ramp_1_l_01_01.csv | 坡道1左侧试次1子试次1 | IMU (6通道) / GON (sagittal角度) |
| ramp_1_l_02_01.csv | 坡道1左侧试次2子试次1 | IMU / GON |
| ramp_1_r_01_01.csv | 坡道1右侧试次1子试次1 | IMU / GON |
| ramp_2_l_01_01.csv | 坡道2左侧试次1子试次1 | IMU / GON |
| ... | ... | ... |

**训练数据划分**：

- 默认训练集比例：80%（通过`--train_ratio`参数配置）
- 默认测试集比例：20%
- 数据划分方法：随机打乱后按比例分割，确保训练集和测试集的样本随机性

### 基准数据集（评估与对比）

本项目使用Goniometer测量数据作为基准真值（Ground Truth）进行模型评估。Goniometer数据存储于`/home/yeqhn/Desktop/TCN/data_processed/ab06_ramp_csv/gon/`目录。

**基准数据特征**：

- 包含多个关节角度测量通道，覆盖髋关节和膝关节的多个自由度
- 模型自动筛选包含"sagittal"关键字的特征列作为评估目标
- 采样率与IMU数据同步（200Hz）

**模型评估指标计算**：

使用MSE、MAE和RMSE指标对比模型预测值与Goniometer基准值之间的误差。这些指标直接反映模型在真实关节角度测量任务中的性能表现。

---

## 补充说明

### 环境依赖

```
torch >= 1.8.0
numpy >= 1.19.0
pandas >= 1.0.0
```

可选依赖：CUDA支持用于GPU加速训练。

### 快速启动

```bash
python prediction_hip_test.py --epochs 50 --batch_size 32 --lr 1e-3 --cuda
```

### 数据目录结构

```
/home/yeqhn/Desktop/TCN/data_processed/
└── ab06_ramp_csv/
    ├── imu/          # IMU传感器数据目录
    │   ├── ramp_1_l_01_01.csv
    │   ├── ramp_1_l_02_01.csv
    │   ├── ramp_1_r_01_01.csv
    │   └── ...        # 其他IMU数据文件
    └── gon/           # Goniometer参考数据目录
        ├── ramp_1_l_01_01.csv
        ├── ramp_1_l_02_01.csv
        ├── ramp_1_r_01_01.csv
        └── ...        # 其他GON数据文件
```

### 关键命令行参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--data_path` | `/home/yeqhn/Desktop/TCN/data_processed` | 数据集根目录 |
| `--train_ratio` | 0.8 | 训练集比例 |
| `--window_seconds` | 1.0 | 输入窗口时长（秒） |
| `--predict_ms` | 100 | 预测时域（毫秒） |
| `--sample_rate` | 200 | 采样率（Hz） |
| `--epochs` | 50 | 训练轮次 |
| `--batch_size` | 32 | 批次大小 |
| `--lr` | 1e-3 | 初始学习率 |
| `--nhid` | 64 | 每层隐藏单元数 |
| `--levels` | 6 | TCN层级数 |
| `--ksize` | 5 | 卷积核大小 |
| `--cuda` | False | 启用GPU训练 |
| `--save_path` | `TCN/prediction_hip_test/results/tcn_model.pt` | 模型保存路径 |
| `--norm_path` | `TCN/prediction_hip_test/results/normalization.pt` | 归一化参数保存路径 |

### 模型输出

训练完成后，模型权重保存至`--save_path`指定路径，归一化参数保存至`--norm_path`指定路径。归一化参数文件包含输入输出数据的均值和标准差，用于推理阶段的数据反归一化。

### 输出文件说明

**模型权重文件**（`tcn_model.pt`）包含：

- 模型结构参数（input_size、output_size、channel_sizes、kernel_size、dropout）
- 模型权重和偏置（state_dict）
- 优化器状态（optimizer_state_dict）
- 训练指标（loss、mse_denorm、rmse_denorm）

**归一化参数文件**（`normalization.pt`）包含：

- input_mean / input_std：输入数据均值和标准差
- output_mean / output_std：输出数据均值和标准差
- imu_features：IMU特征名称列表
- gon_features：GON特征名称列表
- window_samples / predict_samples：滑动窗口配置
- sample_rate：采样率