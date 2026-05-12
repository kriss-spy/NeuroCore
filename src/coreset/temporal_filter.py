import numpy as np
from collections import defaultdict
from src.data_utils import extract_single_arm_action

def compute_temporal_scores(dataset) -> dict[int, float]:
    """
    Computes predictive coding / temporal variance score for each episode.
    Based on the idea that the brain ignores predictable sensory input.
    High prediction error (approximated by sequential action delta) = high value.
    """
    episode_actions = defaultdict(list)
    for i in range(len(dataset)):
        item = dataset[i]
        ep = int(item["episode_index"])
        action = extract_single_arm_action(item["action"])
        episode_actions[ep].append(np.array(action, dtype=np.float32))

    scores = {}
    for ep, actions in episode_actions.items():
        acts = np.stack(actions)
        # Compute delta_t = ||a_t - a_{t-1}||_2
        deltas = np.linalg.norm(acts[1:] - acts[:-1], axis=1)
        # The score is the average frame-to-frame change (action complexity)
        scores[ep] = float(np.mean(deltas))
        
    return scores
