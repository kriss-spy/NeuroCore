"""Build a Colab-ready master notebook."""
import json
import os

CELLS = []

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}

def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [text]}

# 0. Title
CELLS.append(md(
    "# NeuroCore: Brain-Inspired Coreset Selection for Lightweight VLA\n\n"
    "> Course project: Brain-inspired coreset selection for lightweight "
    "Vision-Language-Action (VLA) robotic arm action prediction.\n\n"
    "**Dataset**: ALOHA Sim Transfer Cube (Human Demonstrations) — 50 episodes, ~20 000 frames  \n"
    "**Task**: Predict single-arm 7-DoF actions from frozen ResNet-18 visual features  \n"
    "**Environment**: Local (Jupyter) or Google Colab"
))

# 1. Colab Setup
CELLS.append(md(
    "## Colab Setup (run first if on Google Colab)\n\n"
    "If you are running this notebook on **Google Colab**, execute the cell below to clone the repository "
    "and install dependencies. If running locally, you can skip this cell."
))

CELLS.append(code(
    "import os\n"
    "import sys\n\n"
    "IN_COLAB = 'google.colab' in sys.modules\n"
    "if IN_COLAB:\n"
    "    print('Running in Google Colab — setting up environment...')\n"
    "    if not os.path.exists('NeuroCore'):\n"
    "        !git clone https://github.com/YOUR_USERNAME/NeuroCore.git\n"
    "    %cd NeuroCore\n"
    "    !pip install -q -r requirements.txt\n"
    "    print('Setup complete. Working directory:', os.getcwd())\n"
    "else:\n"
    "    print('Running locally — skipping Colab setup.')\n"
    "    if os.path.basename(os.getcwd()) == 'notebooks':\n"
    "        os.chdir('..')\n"
    "        print('Changed to project root:', os.getcwd())\n"
))

# 2. Imports & Config
CELLS.append(md("## 1. Imports & Configuration"))

CELLS.append(code(
    "import os\n"
    "import sys\n"
    "import json\n\n"
    "project_root = os.path.abspath('.') if os.path.exists('src') else os.path.abspath('..')\n"
    "sys.path.insert(0, project_root)\n\n"
    "import numpy as np\n"
    "import torch\n"
    "import matplotlib.pyplot as plt\n\n"
    "SEED = 42\n"
    "np.random.seed(SEED)\n"
    "torch.manual_seed(SEED)\n\n"
    "RESULTS_DIR = os.path.join(project_root, 'results')\n"
    "FEATURE_PATH = os.path.join(RESULTS_DIR, 'features_resnet18.pt')\n"
    "BASELINE_METRICS_PATH = os.path.join(RESULTS_DIR, 'baseline_metrics.json')\n"
    "CORESET_PATH = os.path.join(RESULTS_DIR, 'coreset_selection.json')\n"
    "CORESET_METRICS_PATH = os.path.join(RESULTS_DIR, 'coreset', 'coreset_metrics.json')\n\n"
    "RUN_TRAINING = True\n"
    "EXTRACT_FEATURES = False\n\n"
    "plt.rcParams['figure.figsize'] = (12, 5)\n"
    "plt.rcParams['figure.dpi'] = 100\n\n"
    "print('CUDA available:', torch.cuda.is_available())\n"
    "print('Project root:', project_root)\n"
    "print('Feature path:', FEATURE_PATH)\n"
    "print('Feature file exists:', os.path.exists(FEATURE_PATH))\n"
))

# 3. Feature Extraction
CELLS.append(md(
    "## 2. Feature Extraction (one-time)\n\n"
    "Visual features are extracted **offline** using a frozen ResNet-18. "
    "Features are cached to disk so extraction is a one-time cost.\n\n"
    "> **Note**: If you cloned this repository, the cached features should already be present. "
    "If not, set `EXTRACT_FEATURES = True` below (requires LeRobot dataset with images)."
))

CELLS.append(code(
    "if EXTRACT_FEATURES or not os.path.exists(FEATURE_PATH):\n"
    "    print('Extracting ResNet-18 features...')\n"
    "    from src.feature_extractor import extract_and_cache_features\n"
    "    features_dict = extract_and_cache_features(FEATURE_PATH)\n"
    "else:\n"
    "    print('Loading cached features...')\n"
    "    features_dict = torch.load(FEATURE_PATH, weights_only=True)\n"
    "    features_dict = {\n"
    "        (int(k[0]), int(k[1])): v\n"
    "        for k, v in features_dict.items()\n"
    "    }\n"
    "print(f'Loaded {len(features_dict)} features, dim={list(features_dict.values())[0].shape[0]}')\n"
))

# 4. Dataset
CELLS.append(md(
    "## 3. Dataset\n\n"
    "We use the **ALOHA Sim Transfer Cube (Human Demonstrations)** dataset from Hugging Face LeRobot.\n\n"
    "- **Episodes**: 50 successful human demonstrations\n"
    "- **Frames**: ~20 000 total (400 frames per episode)\n"
    "- **Actions**: 14-DoF joint actions (we use the first 7 dimensions)\n\n"
    "For this project, we reduce to a single camera view and single-arm 7-DoF actions."
))

CELLS.append(code(
    "from src.data_utils import load_aloha_dataset\n\n"
    "dataset = load_aloha_dataset()\n"
    "num_frames = len(dataset)\n"
    "episodes = [int(item['episode_index']) for item in dataset]\n"
    "num_episodes = max(episodes) + 1\n"
    "frames_per_episode = num_frames // num_episodes\n\n"
    "print(f'Total frames: {num_frames}')\n"
    "print(f'Total episodes: {num_episodes}')\n"
    "print(f'Frames per episode: {frames_per_episode}')\n"
    "print(f'Action dimensions: {len(dataset[0][\"action\"])} (using first 7 for single arm)')\n\n"
    "missing = 0\n"
    "for i in range(len(dataset)):\n"
    "    item = dataset[i]\n"
    "    key = (int(item['episode_index']), int(item['frame_index']))\n"
    "    if key not in features_dict:\n"
    "        missing += 1\n"
    "print(f'Frames without cached features: {missing}')\n"
))

# 5. Baseline
CELLS.append(md(
    "## 4. Baseline — Random 10% Sampling\n\n"
    "Our baseline randomly selects 5 episodes (10% of 50) and trains a lightweight MLP:\n\n"
    "- **Architecture**: 512 → 256 → 128 → 7\n"
    "- **Loss**: Mean Squared Error (MSE)\n"
    "- **Optimizer**: Adam (lr = 1e-3)\n"
    "- **Epochs**: 50  |  **Batch size**: 32\n\n"
    "The model is evaluated on the remaining 45 episodes."
))

CELLS.append(code(
    "from src.baseline import run_baseline, BaselineConfig\n\n"
    "if RUN_TRAINING or not os.path.exists(BASELINE_METRICS_PATH):\n"
    "    print('Training baseline MLP on random 5 episodes...')\n"
    "    baseline_metrics = run_baseline(\n"
    "        feature_path=FEATURE_PATH,\n"
    "        config=BaselineConfig(seed=SEED),\n"
    "        save_dir=RESULTS_DIR,\n"
    "    )\n"
    "    print(f'Baseline MSE: {baseline_metrics[\"mse\"]:.6f}')\n"
    "else:\n"
    "    print('Loading cached baseline metrics...')\n"
    "    with open(BASELINE_METRICS_PATH, 'r') as f:\n"
    "        baseline_metrics = json.load(f)\n"
    "    print(f'Baseline MSE: {baseline_metrics[\"mse\"]:.6f}')\n\n"
    "print(f'Train episodes: {baseline_metrics[\"train_episodes\"]}')\n"
))

CELLS.append(code(
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "ax1.plot(baseline_metrics['train_losses'], label='Train', color='blue')\n"
    "ax1.plot(baseline_metrics['val_losses'], label='Validation', color='orange')\n"
    "ax1.set_title('Baseline Loss Curves')\n"
    "ax1.set_xlabel('Epoch')\n"
    "ax1.set_ylabel('MSE')\n"
    "ax1.legend()\n"
    "ax1.grid(True, alpha=0.3)\n\n"
    "joints = list(range(7))\n"
    "ax2.bar(joints, baseline_metrics['per_joint_mse'], color='steelblue', edgecolor='black')\n"
    "ax2.set_title('Baseline Per-Joint Test MSE')\n"
    "ax2.set_xlabel('Joint Index')\n"
    "ax2.set_ylabel('MSE')\n"
    "ax2.grid(True, alpha=0.3)\n\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# 6. Coreset Algorithm
CELLS.append(md(
    "## 5. Coreset Selection Algorithm\n\n"
    "We replace random sampling with a brain-inspired pruning algorithm that fuses two filters:\n\n"
    "### 5.1 Temporal Filter — Predictive Coding\n"
    "For each episode, compute the mean frame-to-frame action change:\n"
    "$$\\delta_t = \\|a_t - a_{t-1}\\|_2$$\n"
    "Episodes with high motion complexity score higher.\n\n"
    "### 5.2 Distributional Filter — RAS\n"
    "Cluster all frame-level ResNet-18 features (K-Means, k=15). "
    "For each episode, compute entropy of cluster occupancy × coverage ratio.\n\n"
    "### 5.3 Unified Selection\n"
    "Normalize both scores to [0,1] and fuse with weight $\\alpha$:\n"
    "$$S_{\\text{final}} = \\alpha \\cdot \\tilde{S}_{\\text{temp}} + (1 - \\alpha) \\cdot \\tilde{S}_{\\text{dist}}$$\n\n"
    "After ablation, we use **$\\alpha = 0.75$** (temporal-heavy, diversity-tempered)."
))

CELLS.append(code(
    "from src.coreset.select import select_coreset\n\n"
    "if RUN_TRAINING or not os.path.exists(CORESET_PATH):\n"
    "    print('Running coreset selection (alpha=0.75)...')\n"
    "    selected_episodes, final_scores, temp_norm, dist_norm = select_coreset(\n"
    "        dataset, features_dict, k=5, alpha=0.75, save_path=CORESET_PATH\n"
    "    )\n"
    "    print(f'Selected episodes: {selected_episodes}')\n"
    "else:\n"
    "    print('Loading cached coreset selection...')\n"
    "    with open(CORESET_PATH, 'r') as f:\n"
    "        coreset_data = json.load(f)\n"
    "    selected_episodes = coreset_data['selected_episodes']\n"
    "    final_scores = {int(k): v for k, v in coreset_data['scores'].items()}\n"
    "    temp_norm = {int(k): v for k, v in coreset_data['temporal_scores'].items()}\n"
    "    dist_norm = {int(k): v for k, v in coreset_data['distributional_scores'].items()}\n"
    "    print(f'Selected episodes: {selected_episodes}')\n"
))

CELLS.append(code(
    "episodes = sorted(list(final_scores.keys()))\n"
    "t_vals = [temp_norm[e] for e in episodes]\n"
    "d_vals = [dist_norm.get(e, 0) for e in episodes]\n"
    "f_vals = [final_scores[e] for e in episodes]\n"
    "colors = ['red' if e in selected_episodes else 'gray' for e in episodes]\n\n"
    "fig, axes = plt.subplots(1, 3, figsize=(16, 4))\n\n"
    "axes[0].bar(episodes, t_vals, color=colors, edgecolor='black', alpha=0.8)\n"
    "axes[0].set_title('Temporal Score (Predictive Coding)')\n"
    "axes[0].set_xlabel('Episode')\n"
    "axes[0].set_ylabel('Normalized Score')\n"
    "axes[0].grid(True, alpha=0.3)\n\n"
    "axes[1].bar(episodes, d_vals, color=colors, edgecolor='black', alpha=0.8)\n"
    "axes[1].set_title('Distributional Score (RAS)')\n"
    "axes[1].set_xlabel('Episode')\n"
    "axes[1].set_ylabel('Normalized Score')\n"
    "axes[1].grid(True, alpha=0.3)\n\n"
    "axes[2].bar(episodes, f_vals, color=colors, edgecolor='black', alpha=0.8)\n"
    "axes[2].set_title('Combined Score (alpha=0.75) — Top-5 in Red')\n"
    "axes[2].set_xlabel('Episode')\n"
    "axes[2].set_ylabel('Normalized Score')\n"
    "axes[2].grid(True, alpha=0.3)\n\n"
    "plt.tight_layout()\n"
    "plt.show()\n"
))

# 7. Validation
CELLS.append(md(
    "## 6. Validation — Coreset vs. Baseline\n\n"
    "We retrain the **identical MLP architecture** on the selected coreset episodes and evaluate on the "
    "**same held-out 45 episodes** as the baseline. This isolates the effect of data selection from model architecture."
))

CELLS.append(code(
    "from src.validate import run_validation\n\n"
    "if RUN_TRAINING or not os.path.exists(CORESET_METRICS_PATH):\n"
    "    print('Running coreset validation...')\n"
    "    validation_results = run_validation(\n"
    "        coreset_path=CORESET_PATH,\n"
    "        feature_path=FEATURE_PATH,\n"
    "        baseline_metrics_path=BASELINE_METRICS_PATH,\n"
    "        save_dir=os.path.join(RESULTS_DIR, 'coreset'),\n"
    "    )\n"
    "    coreset_metrics = validation_results['coreset_metrics']\n"
    "else:\n"
    "    print('Loading cached validation results...')\n"
    "    with open(CORESET_METRICS_PATH, 'r') as f:\n"
    "        comparison = json.load(f)\n"
    "    with open(os.path.join(RESULTS_DIR, 'coreset', 'baseline_metrics.json'), 'r') as f:\n"
    "        coreset_metrics = json.load(f)\n"
    "    print(f'Coreset MSE: {coreset_metrics[\"mse\"]:.6f}')\n"
    "    print(f'Baseline MSE: {comparison[\"baseline_mse\"]:.6f}')\n"
    "    print(f'Improvement: {comparison[\"improvement_percent\"]:.2f}%')\n"
))

CELLS.append(code(
    "with open(BASELINE_METRICS_PATH, 'r') as f:\n"
    "    baseline = json.load(f)\n\n"
    "with open(os.path.join(RESULTS_DIR, 'coreset', 'baseline_metrics.json'), 'r') as f:\n"
    "    coreset = json.load(f)\n\n"
    "fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))\n\n"
    "ax1.plot(baseline['val_losses'], label='Random Baseline', color='gray', linestyle='--', linewidth=2)\n"
    "ax1.plot(coreset['val_losses'], label='Coreset', color='blue', linewidth=2)\n"
    "ax1.set_title('Validation Loss Convergence')\n"
    "ax1.set_xlabel('Epoch')\n"
    "ax1.set_ylabel('MSE')\n"
    "ax1.legend()\n"
    "ax1.grid(True, alpha=0.3)\n\n"
    "joints = list(range(7))\n"
    "width = 0.35\n"
    "ax2.bar([j - width/2 for j in joints], baseline['per_joint_mse'], width,\n"
    "        label='Random Baseline', color='gray', edgecolor='black')\n"
    "ax2.bar([j + width/2 for j in joints], coreset['per_joint_mse'], width,\n"
    "        label='Coreset', color='blue', edgecolor='black')\n"
    "ax2.set_title('Per-Joint Test MSE')\n"
    "ax2.set_xlabel('Joint Index')\n"
    "ax2.set_ylabel('MSE')\n"
    "ax2.legend()\n"
    "ax2.grid(True, alpha=0.3)\n\n"
    "plt.tight_layout()\n"
    "plt.show()\n\n"
    "print('\\n' + '='*50)\n"
    "print(f\"{'Method':<20} {'Test MSE':<15} {'Improvement'}\")\n"
    "print('='*50)\n"
    "print(f\"{'Random Baseline':<20} {baseline['mse']:.6f}     —\")\n"
    "improvement = (baseline['mse'] - coreset['mse']) / baseline['mse'] * 100\n"
    "print(f\"{'Coreset (Ours)':<20} {coreset['mse']:.6f}     +{improvement:.2f}%\")\n"
    "print('='*50)\n"
))

# 8. Discussion & Conclusion
CELLS.append(md(
    "## 7. Discussion & Conclusion\n\n"
    "### Why does the coreset work?\n\n"
    "The optimal $\\alpha = 0.75$ suggests that **temporal action novelty** is the dominant signal for learning, "
    "but must be tempered with **visual diversity** to avoid over-selecting chaotic episodes. "
    "This mirrors neuroscience: predictive coding drives cortical learning, but the RAS modulates which errors reach conscious processing.\n\n"
    "### Limitations\n\n"
    "1. **Dataset size**: 50 episodes is small; random baseline variance is high.\n"
    "2. **Episode-level granularity**: Frame-level pruning could yield further gains.\n"
    "3. **Single view, single arm**: Multi-modal fusion is left for future work.\n\n"
    "### Definition of Redundancy\n\n"
    "We define redundancy at two timescales:\n\n"
    "1. **Temporal redundancy**: A frame is redundant if $\\delta_t = \\|a_t - a_{t-1}\\|_2$ is small — the action is predictable from the previous frame.\n"
    "2. **Distributional redundancy**: A frame is redundant if it belongs to a visual cluster already well-represented in the selected set.\n\n"
    "This definition is brain-inspired because it explicitly discards data that the brain's own filtering mechanisms (predictive coding + RAS) would \"explain away.\"\n\n"
    "### Reproduction\n\n"
    "```bash\n"
    "python -m src.baseline        # Run random baseline\n"
    "python -m src.coreset.select  # Run coreset selection\n"
    "python -m src.validate        # Validate and compare\n"
    "```"
))

# Build notebook
nb = {
    "cells": CELLS,
    "metadata": {
        "kernelspec": {"display_name": ".venv", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.0"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}

os.makedirs("notebooks", exist_ok=True)
with open("notebooks/NeuroCore_full.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print(f"Colab-ready notebook written to notebooks/NeuroCore_full.ipynb ({len(CELLS)} cells)")
