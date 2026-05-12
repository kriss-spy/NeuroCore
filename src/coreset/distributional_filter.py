import numpy as np
import torch
from sklearn.cluster import KMeans
from collections import defaultdict
from typing import Dict, Tuple

def compute_distributional_scores(features_dict: Dict[Tuple[int, int], torch.Tensor], n_clusters: int = 15, random_state: int = 42) -> dict[int, float]:
    """
    Simulates RAS (Reticular Activating System) to filter noise.
    Clusters visual features and defines 'utility' as covering a diverse set of states.
    Calculated via the entropy of an episode's cluster occupancy.
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
