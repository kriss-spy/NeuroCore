import json
import os
from typing import Dict, List, Tuple
import torch

from src.coreset.temporal_filter import compute_temporal_scores
from src.coreset.distributional_filter import compute_distributional_scores

def normalize_dict(d: dict[int, float]) -> dict[int, float]:
    vals = list(d.values())
    if not vals:
        return d
    vmin, vmax = min(vals), max(vals)
    span = vmax - vmin if vmax > vmin else 1.0
    return {k: (v - vmin) / span for k, v in d.items()}

def select_coreset(dataset, features_dict: Dict[Tuple[int, int], torch.Tensor], k: int = 5, alpha: float = 0.5) -> Tuple[List[int], dict, dict, dict]:
    """
    Combines the Temporal Prediction score (Alpha weight) with the Distributional RAS Score (1-Alpha weight).
    Selects top K episodes.
    """
    print(f"Computing temporal scores (Alpha={alpha})...")
    temp_scores = compute_temporal_scores(dataset)
    
    print("Computing distributional scores...")
    dist_scores = compute_distributional_scores(features_dict)
    
    temp_norm = normalize_dict(temp_scores)
    dist_norm = normalize_dict(dist_scores)
    
    final_scores = {}
    for ep in temp_scores.keys():
        # A true brain-inspired fusion mechanism
        final_scores[ep] = (alpha * temp_norm[ep]) + ((1 - alpha) * dist_norm.get(ep, 0.0))
        
    # Sort descending
    sorted_eps = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    selected_episodes = [ep for ep, score in sorted_eps[:k]]
    
    os.makedirs("results", exist_ok=True)
    out_path = "results/coreset_selection.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(selected_episodes, f, indent=2)
        
    return selected_episodes, final_scores, temp_norm, dist_norm
