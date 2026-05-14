import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Dict, Tuple

def compute_distributional_scores(features_dict: Dict[Tuple[int, int], torch.Tensor], n_clusters: int = 15, random_state: int = 42) -> dict[int, float]:
    """
    RAS 启发的分布冗余度评分（RAS-inspired Distributional Scoring）。

    脑启发机制：
    网状激活系统（Reticular Activating System, RAS）是位于脑干的神经核团网络，
    负责调节觉醒、注意力和信息过滤。RAS 的核心功能是抑制背景噪音和无关刺激，
    只将具有高信息效用的信号传递到高级皮层进行处理。在大量同质刺激中，RAS 会
    降低重复信号的传递权重，而放大新奇或行为相关的刺激。

    在机器人数据分布层面，我们使用 K-Means 聚类将视觉特征空间离散化为若干
    "任务状态簇"。对于每个片段：
    - 如果片段的帧集中在少数几个簇中，说明机器人在重复相似的视觉配置，
      类似于被 RAS 过滤掉的"背景噪音"，分布冗余度高。
    - 如果片段覆盖了更多不同的簇，说明机器人探索了多样化的任务状态，
      类似于被 RAS "注意到"的高效用时刻，分布冗余度低。

    我们通过香农熵（Entropy）乘以覆盖率（Coverage Ratio）来量化这种多样性：
        H(p_e) = -sum(p_e(c) * log(p_e(c)))
        Coverage = |{c : p_e(c) > 0}| / C
        S_dist(e) = H(p_e) * Coverage

    对应脑机制：Reticular Activating System（RAS）
    """
    keys = list(features_dict.keys())
    # Stack features into numpy array
    X = np.stack([features_dict[k].numpy() for k in keys])
    
    # Run K-Means across the entire dataset latent space
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init="auto")
    labels = kmeans.fit_predict(X)
    
    # Assign labels back to episodes
    episode_clusters = defaultdict(list)
    for i, (ep_idx, _) in enumerate(keys):
        episode_clusters[int(ep_idx)].append(labels[i])
        
    scores = {}
    for ep, clusters in episode_clusters.items():
        # Count frequency of each cluster hit by this episode
        unique, counts = np.unique(clusters, return_counts=True)
        
        # Entropy computation for diversity
        probs = counts / counts.sum()
        entropy = -np.sum(probs * np.log(probs + 1e-9))
        
        # Coverage penalty (favor episodes that hit many unique task stages)
        coverage_ratio = len(unique) / n_clusters
        
        scores[ep] = float(entropy * coverage_ratio)
        
    return scores
