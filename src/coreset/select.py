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


def select_coreset(
    dataset,
    features_dict: Dict[Tuple[int, int], torch.Tensor],
    k: int = 5,
    alpha: float = 0.75,
    save_path: str = "results/coreset_selection.json",
) -> Tuple[List[int], dict, dict, dict]:
    """
    Combines the Temporal Prediction score (Alpha weight) with the
    Distributional RAS Score (1-Alpha weight). Selects top K episodes.

    Returns:
        selected_episodes: list of episode indices
        final_scores: dict episode -> normalized combined score
        temp_norm: dict episode -> normalized temporal score
        dist_norm: dict episode -> normalized distributional score
    """
    print(f"Computing temporal scores (Alpha={alpha})...")
    temp_scores = compute_temporal_scores(dataset)

    print("Computing distributional scores...")
    dist_scores = compute_distributional_scores(features_dict)

    temp_norm = normalize_dict(temp_scores)
    dist_norm = normalize_dict(dist_scores)

    final_scores = {}
    for ep in temp_scores.keys():
        final_scores[ep] = (alpha * temp_norm[ep]) + (
            (1 - alpha) * dist_norm.get(ep, 0.0)
        )

    # Sort descending
    sorted_eps = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    selected_episodes = [ep for ep, score in sorted_eps[:k]]

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "selected_episodes": selected_episodes,
                "scores": {str(k): v for k, v in final_scores.items()},
                "temporal_scores": {str(k): v for k, v in temp_scores.items()},
                "distributional_scores": {str(k): v for k, v in dist_scores.items()},
            },
            f,
            indent=2,
        )

    return selected_episodes, final_scores, temp_norm, dist_norm


if __name__ == "__main__":
    import sys

    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    from src.data_utils import load_aloha_dataset

    dataset = load_aloha_dataset()
    features_dict = torch.load("results/features_resnet18.pt", weights_only=True)
    selected, final_scores, temp_norm, dist_norm = select_coreset(
        dataset, features_dict, k=5, alpha=0.75
    )
    print(f"Selected top-5 episodes: {selected}")
