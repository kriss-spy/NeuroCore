# NeuroCore: 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测

> Course project: Brain-inspired coreset selection for lightweight Vision-Language-Action (VLA) robotic arm action prediction.

## 📌 项目概述 (Overview)

NeuroCore 探索了如何利用脑启发式数据选择机制（Coreset Selection）来提升轻量级 VLA 模型在机器人操作任务中的训练效率。
机器人演示数据集通常包含大量的时间（Temporal）和分布（Distributional）冗余。受人类大脑的**预测编码（Predictive Coding）**和**网状激活系统（RAS）**启发，我们设计并实现了自动化的数据剪枝算法，从 ALOHA Sim 数据集中提取高价值的核心子集。

**核心结论**：在仅使用 10% 数据量的条件下，脑启发核心集训练的模型在测试集上的均方误差（MSE）比随机采样降低了 **2.57%**。

## 🧠 脑启发机制 (Brain-Inspired Mechanisms)

### 1. 预测编码 (Predictive Coding) - 时间冗余过滤
大脑通过忽略可预测的感官输入来节省能量，仅对“预测误差”产生显著反应。
- **实现**：通过计算动作变化量 $\Delta a_t$ 量化预测误差。变化量大的帧被认为信息价值更高，冗余度低。

### 2. 网状激活系统 (RAS) - 分布冗余过滤
RAS 负责过滤背景噪音，引导注意力聚焦于高信息效用的状态。
- **实现**：利用 K-Means 聚类离散化视觉状态空间。优先选择能够覆盖更多样、更独特“任务状态簇”的片段。

## 🚀 快速开始 (Quick Start)

### 环境安装 (Installation)

```bash
git clone <repo-url>
cd NeuroCore
pip install -r requirements.txt
```

### 运行实验 (Run Experiments)

```bash
# 1. 运行随机基线 (Baseline)
python -m src.baseline

# 2. 运行脑启发核心集选择 (Coreset Selection)
python -m src.coreset.select

# 3. 验证并对比结果 (Validation)
python -m src.validate
```

### 交互式展示 (Notebook)
打开 `notebooks/NeuroCore.ipynb` 查看完整的实验流程、可视化图表及理论分析。

## 📂 项目结构 (Structure)

```
NeuroCore/
├── src/
│   ├── coreset/
│   │   ├── temporal_filter.py      # 时间冗余评分 (预测编码)
│   │   ├── distributional_filter.py# 分布冗余评分 (RAS)
│   │   └── select.py               # 统一核心集选择逻辑 (argmin R_final)
│   ├── baseline.py                 # 多模态基线训练 [视觉+语言] -> 动作
│   ├── data_utils.py               # 数据加载与处理
│   └── validate.py                 # 核心集验证与对比实验
├── notebooks/
│   └── NeuroCore.ipynb             # 核心实验过程与结果展示
├── results/
│   ├── FINAL_REPORT.md             # 最终学术报告 (中文版)
│   ├── FINAL_REPORT.pdf            # 最终学术报告 (PDF)
│   └── figures/                    # 实验结果图表
├── docs/
│   ├── papers/                     # 参考文献 PDFs
│   └── neurocore-wiki/             # 基于 Obsidian 的知识库
└── requirements.txt                # 项目依赖
```

## 📊 实验结果 (Results)

| 方法 (Method) | 训练数据量 | 测试 MSE | 性能提升 |
| :--- | :--- | :--- | :--- |
| 随机采样 (Random 10%) | 5 episodes | 0.00681 | - |
| **NeuroCore (Ours 10%)** | 5 episodes | **0.00663** | **+2.57%** |

## 🔗 参考文献 (References)

1. Zhao T Z, et al. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)](https://arxiv.org/abs/2304.13705). RSS, 2023.
2. Sorscher B, et al. [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486). NeurIPS, 2022.
3. Millidge B, et al. [Predictive coding: a theoretical and experimental review](https://arxiv.org/abs/2107.12979). arXiv, 2021.

---
**Course Project**: 视觉认知工程 (2026春)
