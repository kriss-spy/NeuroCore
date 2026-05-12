# NeuroCore: Brain-Inspired Coreset Selection for Lightweight VLA

> Course project: Brain-inspired coreset selection for lightweight Vision-Language-Action (VLA) robotic arm action prediction.

## Overview

NeuroCore explores how brain-inspired data selection mechanisms can improve the efficiency of training lightweight VLA models for robotic manipulation. Real-world robot demonstration datasets contain massive temporal and distributional redundancy, making full-dataset training computationally wasteful. By drawing inspiration from the human brain's **predictive coding** and **reticular activating system (RAS)**, we design automated data-pruning algorithms to extract high-value coresets from the ALOHA Sim Transfer Cube dataset.

**Key insight**: The human brain processes enormous sensory data streams using only ~20 watts of power, thanks to efficient filtering mechanisms. We operationalize these mechanisms to prune redundant robot demonstration data and train better models with less compute.

## Architecture

The project consists of three stages:

### 1. Baseline
- Randomly sample 10% of trajectories from the ALOHA Sim dataset
- Extract visual features with **frozen ResNet-18** or **CLIP** (offline, no training)
- Build regression dataset: `[visual_features, language_instruction] → [7-DoF arm action]`
- Train a lightweight **MLP** and report **Mean Squared Error (MSE)**

### 2. Brain-Inspired Coreset Selection
Replace random sampling with an automated pruning algorithm:

- **Temporal Redundancy Filtering** (Predictive Coding-inspired): Filter frames based on continuous action variance—keep only frames where the action changes significantly
- **Distributional Redundancy Filtering** (RAS-inspired): Cluster visual features and select representative samples to ensure coverage of the action distribution

Target: Select the "most valuable" 10% subset that preserves or improves model performance.

### 3. Validation
- Retrain the same MLP architecture on the selected 10% coreset
- Report MSE and compare against the random baseline
- Demonstrate that high-quality data subsets yield better model performance than random sampling

## Dataset

**ALOHA Sim Transfer Cube (Human Demonstrations)** from [Hugging Face LeRobot](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human)

- 50 successful human demonstration episodes
- Multi-view camera images + 14-DoF joint actions (7-DoF per arm)
- ~200 MB total, suitable for laptop training
- Can reduce to single camera view + single-arm 7-DoF actions for simplicity

```python
from datasets import load_dataset

dataset = load_dataset("lerobot/aloha_sim_transfer_cube_human")
```

## Quick Start

```bash
# Clone the repository
git clone <repo-url>
cd NeuroCore

# Install dependencies
pip install -r requirements.txt

# Run baseline
python src/baseline.py

# Run coreset selection
python src/coreset/select.py

# Validate
python src/validate.py
```

## Project Structure

```
NeuroCore/
├── README.md                    # This file
├── AGENTS.md                    # Agent instructions for OpenCode
├── PLAN.md                      # Execution plan and checklist
├── requirements.txt             # Python dependencies
├── .env                         # Wiki path configuration
├── src/
│   ├── data_utils.py           # Dataset loading and action extraction
│   ├── feature_extractor.py    # Frozen ResNet-18 feature extraction
│   ├── baseline.py             # Random sampling baseline + MLP
│   └── coreset/
│       ├── __init__.py
│       ├── select.py           # Unified coreset selection
│       ├── temporal_filter.py  # Predictive coding–inspired scoring
│       └── distributional_filter.py  # RAS-inspired clustering scoring
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_coreset_selection.ipynb
│   └── 05_coreset_validation.ipynb
├── docs/
│   ├── papers/                 # Reference PDFs
│   └── neurocore-wiki/         # Project knowledge base (Obsidian)
└── results/                    # Cached features, checkpoints, metrics
```

## Brain-Inspired Mechanisms

### Predictive Coding (Temporal Filtering)
The brain constantly generates predictions about incoming sensory data. Only prediction errors—moments when reality diverges from expectation—trigger strong neural activation. We operationalize this by:
- Computing action variance across temporal windows
- Filtering out "idle frames" where the robot is stationary or repeating the same action
- Retaining frames with high action novelty

### Reticular Activating System (Distributional Filtering)
The RAS filters background noise and focuses attention on high-information-utility moments. We operationalize this by:
- Clustering visual features to identify action modes
- Selecting diverse representatives from each cluster
- Ensuring coverage of rare but important actions (e.g., grasping, releasing)

## Expected Results

| Method | Data Used | Expected MSE |
|--------|-----------|--------------|
| Random Baseline | 10% random | Higher |
| Coreset (Ours) | 10% selected | Lower |

*Hypothesis: Coreset selection achieves lower MSE than random sampling with the same data budget, demonstrating that data quality matters more than quantity.*

## References

1. **[ACT]** Zhao T Z, et al. [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705). RSS, 2023.
2. **[Data Pruning]** Sorscher B, et al. [Beyond neural scaling laws: beating power law scaling via data pruning](https://arxiv.org/abs/2206.14486). NeurIPS, 2022.
3. **[OpenVLA]** Kim M J, et al. [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246). arXiv, 2024.
4. **[Predictive Coding]** Millidge B, et al. [Predictive coding: a theoretical and experimental review](https://arxiv.org/abs/2107.12979). arXiv, 2021.

## License

This is a course project for educational purposes.

---

**Original Assignment**: 基于脑启发核心集选择的轻量级 VLA 机械臂动作预测 (Brain-Inspired Coreset Selection for Lightweight VLA Robotic Arm Action Prediction)
