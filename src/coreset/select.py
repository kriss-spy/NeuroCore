from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from src.coreset.distributional_filter import (
    DistributionalFilterConfig,
    compute_distributional_scores,
)
from src.coreset.temporal_filter import TemporalFilterConfig, compute_temporal_scores


@dataclass
class CoresetConfig:
    """
    Configuration for coreset selection.

    k: Number of episodes to select.
    alpha: Weight for temporal score (0.0 = only distributional, 1.0 = only temporal).
    """

    k: int = 5
    alpha: float = 0.5


def _normalize(scores: Dict[int, float]) -> Dict[int, float]:
    values = np.array(list(scores.values()), dtype=np.float32)
    if np.allclose(values.max(), values.min()):
        return {k: 0.0 for k in scores}
    normalized = (values - values.min()) / (values.max() - values.min())
    return {k: float(v) for k, v in zip(scores.keys(), normalized)}


def select_coreset(
    features_path: str,
    config: CoresetConfig | None = None,
    temporal_cfg: TemporalFilterConfig | None = None,
    distributional_cfg: DistributionalFilterConfig | None = None,
) -> Dict[str, object]:
    """
    Select top-k episodes using combined temporal + distributional scores.

    Returns a dictionary with scores and selected episodes.
    """

    cfg = config or CoresetConfig()
    temporal_cfg = temporal_cfg or TemporalFilterConfig()
    distributional_cfg = distributional_cfg or DistributionalFilterConfig()

    features = torch.load(features_path, weights_only=True)

    temporal_scores = compute_temporal_scores(temporal_cfg)
    distribution_scores = compute_distributional_scores(features, distributional_cfg)

    temporal_norm = _normalize(temporal_scores)
    distribution_norm = _normalize(distribution_scores)

    combined: Dict[int, float] = {}
    for ep in temporal_norm.keys():
        combined[ep] = cfg.alpha * temporal_norm[ep] + (
            1 - cfg.alpha
        ) * distribution_norm.get(ep, 0.0)

    selected = sorted(combined.items(), key=lambda kv: kv[1], reverse=True)[: cfg.k]
    selected_ids = [ep for ep, _ in selected]

    results = {
        "selected_episodes": selected_ids,
        "scores": combined,
        "temporal_scores": temporal_scores,
        "distributional_scores": distribution_scores,
    }

    with open("results/coreset_selection.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results
