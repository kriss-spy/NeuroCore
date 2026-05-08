from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from src.data_utils import extract_single_arm_action, load_aloha_dataset


@dataclass
class TemporalFilterConfig:
    """
    Configuration for predictive-coding temporal redundancy scoring.

    window: Number of frames between action comparisons.
    idle_threshold: If set, only deltas above this threshold count as non-idle.
    """

    window: int = 1
    idle_threshold: float | None = None


def compute_temporal_scores(
    config: TemporalFilterConfig | None = None,
) -> Dict[int, float]:
    """
    Compute temporal action-variance scores per episode.

    Returns:
        A dict mapping episode index -> temporal score.
    """

    cfg = config or TemporalFilterConfig()
    dataset = load_aloha_dataset()
    num_episodes = dataset.num_episodes
    scores: Dict[int, List[float]] = {ep: [] for ep in range(num_episodes)}

    # Collect action sequences per episode in order
    episode_actions: Dict[int, List[np.ndarray]] = {
        ep: [] for ep in range(num_episodes)
    }
    for idx in range(len(dataset)):
        item = dataset[idx]
        ep = int(item["episode_index"])
        action = extract_single_arm_action(item["action"])
        episode_actions[ep].append(np.array(action, dtype=np.float32))

    for ep, actions in episode_actions.items():
        if len(actions) <= cfg.window:
            scores[ep] = []
            continue
        for t in range(cfg.window, len(actions)):
            delta = np.linalg.norm(actions[t] - actions[t - cfg.window])
            if cfg.idle_threshold is None or delta > cfg.idle_threshold:
                scores[ep].append(delta)

    # Aggregate to episode-level score
    episode_scores: Dict[int, float] = {}
    for ep, deltas in scores.items():
        if len(deltas) == 0:
            episode_scores[ep] = 0.0
        else:
            episode_scores[ep] = float(np.mean(deltas))

    return episode_scores
