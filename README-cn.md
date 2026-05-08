# NeuroCore: 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

> 课程设计项目：基于脑启发核心集选择的轻量级视觉-语言-动作（VLA）机械臂动作预测。

## 概述

NeuroCore 探索如何利用脑启发的数据选择机制来提高训练轻量级 VLA 模型进行机器人操作的效率。真实机器人演示数据集包含大量的时间分布冗余，使得全数据集训练在计算上非常浪费。通过借鉴人脑的**预测编码**和**网状激活系统（RAS）**，我们设计了自动化的数据剪枝算法，从 ALOHA Sim Transfer Cube 数据集中提取高价值的核心集。

**核心洞察**：人脑仅使用约 20 瓦的功率就能处理巨大的感官数据流，这得益于高效的过滤机制。我们将这些机制应用于剪枝冗余的机器人演示数据，用更少的计算资源训练出更好的模型。

## 架构

该项目包含三个阶段：

### 1. 基准测试（Baseline）
- 从 ALOHA Sim 数据集中随机抽取 10% 的轨迹样本
- 使用**冻结权重的 ResNet-18** 或 **CLIP** 离线提取图像特征（无需训练）
- 构建回归数据集：`[视觉特征, 语言指令] → [7 自由度机械臂动作]`
- 训练一个轻量级的**多层感知机（MLP）**，并报告**均方误差（MSE）**

### 2. 脑启发核心集选择
用自动化的剪枝算法替代随机采样：

- **时间冗余过滤**（预测编码启发）：基于连续动作方差过滤帧——仅保留动作发生显著变化的帧
- **分布冗余过滤**（RAS 启发）：对视觉特征进行聚类并选择代表性样本，以确保覆盖动作分布

目标：选择"最有价值"的 10% 子集，以保留或提升模型性能。

### 3. 验证
- 在选出的 10% 核心集上重新训练相同的 MLP 架构
- 报告 MSE 并与随机基线进行比较
- 证明高质量的数据子集比随机采样带来更好的模型性能

## 数据集

**ALOHA Sim Transfer Cube (Human Demonstrations)** 来自 [Hugging Face LeRobot](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)

- 50 个成功的人类演示轨迹
- 多视角相机图像 + 14 自由度联合动作（单臂 7 自由度）
- 总计约 200 MB，适合在笔记本电脑上训练
- 为简化实验，可仅使用单一相机视角和单臂 7 自由度动作

```python
from datasets import load_dataset

dataset = load_dataset("lerobot/aloha_sim_transfer_cube_human")
```

## 快速开始

```bash
# 克隆仓库
git clone <repo-url>
cd NeuroCore

# 安装依赖（待完成：创建 requirements.txt）
pip install -r requirements.txt

# 运行基准测试
python src/baseline.py

# 运行核心集选择
python src/coreset/select.py

# 验证
python src/validate.py
```

## 项目结构

```
NeuroCore/
├── README.md                    # 本文件
├── README.zh.md                 # 中文版本
├── AGENTS.md                    # OpenCode 代理指令
├── requirements.txt             # Python 依赖（待完成）
├── src/
│   ├── baseline.py             # 随机采样基线
│   ├── coreset/
│   │   ├── select.py           # 核心集选择算法
│   │   └── metrics.py          # 冗余度指标
│   └── validate.py             # 验证与对比
├── docs/
│   ├── papers/                 # 参考论文 PDF
│   └── obsidian-wiki/          # 项目知识库
└── results/                    # 实验输出（待完成）
```

## 脑启发机制

### 预测编码（时间过滤）
大脑不断生成关于传入感官数据的预测。只有预测误差——即现实与预期发生偏离的时刻——才会触发强烈的神经激活。我们通过以下方式实现这一机制：
- 计算时间窗口内的动作方差
- 过滤掉机器人静止或重复相同动作的"空闲帧"
- 保留具有高动作新颖性的帧

### 网状激活系统（分布过滤）
RAS 过滤背景噪音，并将注意力集中在高信息效用的时刻。我们通过以下方式实现这一机制：
- 对视觉特征进行聚类以识别动作模式
- 从每个聚类中选择多样化的代表样本
- 确保覆盖罕见但重要的动作（例如抓取、释放）

## 预期结果

| 方法 | 使用数据 | 预期 MSE |
|------|----------|----------|
| 随机基线 | 10% 随机样本 | 较高 |
| 核心集（本文） | 10% 选择样本 | 较低 |

*假设：在相同数据预算下，核心集选择比随机采样实现更低的 MSE，证明数据质量比数量更重要。*

## 参考文献

1. **[ACT]** Zhao T Z, 等. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705). RSS, 2023.
2. **[数据剪枝]** Sorscher B, 等. [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486). NeurIPS, 2022.
3. **[OpenVLA]** Kim M J, 等. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246). arXiv, 2024.
4. **[预测编码]** Millidge B, 等. [Predictive coding: a theoretical and experimental review](https://arxiv.org/abs/2107.12979). arXiv, 2021.

## 许可协议

本项目为课程设计，仅供教育目的使用。

---

**原始题目**：基于脑启发核心集选择的轻量级 VLA 机械臂动作预测
