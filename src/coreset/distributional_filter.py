from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
from sklearn.cluster import KMeans


@dataclass
class DistributionalFilterConfig:
    """
    Configuration for distributional redundancy scoring.

    clusters: K for k-means.
    random_state: Random seed for KMeans.
    """

    clusters: int = 10
    random_state: int = 42


def compute_distributional_scores(
    features: Dict[Tuple[int, int], np.ndarray] | Dict[Tuple[int, int], object],
    config: DistributionalFilterConfig | None = None,
) -> Dict[int, float]:
    """
    Compute diversity scores for each episode via k-means cluster coverage.

    Returns:
        A dict mapping episode index -> diversity score.
    """

    cfg = config or DistributionalFilterConfig()

    # Flatten features and keep track of episode index
    eps = []
    feats = []
    for (ep_idx, _), feat in features.items():
        eps.append(int(ep_idx))
        feats.append(np.asarray(feat))

    feats = np.vstack(feats)
    kmeans = KMeans(
        n_clusters=cfg.clusters, random_state=cfg.random_state, n_init="auto"
    )
    cluster_ids = kmeans.fit_predict(feats)

    # Compute entropy-based diversity per episode
    scores: Dict[int, float] = {}
    for ep in set(eps):
        ep_clusters = cluster_ids[np.array(eps) == ep]
        if len(ep_clusters) == 0:
            scores[ep] = 0.0
            continue
        hist, _ = np.histogram(ep_clusters, bins=cfg.clusters, range=(0, cfg.clusters))
        probs = hist / np.sum(hist)
        # Shannon entropy
        entropy = -np.sum([p * np.log(p + 1e-8) for p in probs])
        scores[ep] = float(entropy)

    return scores
