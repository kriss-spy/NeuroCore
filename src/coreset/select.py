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
    统一核心集选择：融合预测编码和 RAS 的层级冗余过滤。

    脑启发机制：
    大脑在多个层级上并行过滤信息，形成层级化的信息处理流水线：

    1. 早期感觉阶段（预测编码）：位于较低层级的皮层区域（如初级感觉皮层）
       通过预测编码过滤时间上的可预测信号。如果输入在时间上是可预测的
       （例如机器人静止），预测误差很小，信号在此阶段就被抑制，不会向上传递。

    2. 晚期注意阶段（RAS）：经过早期过滤的信号进入网状激活系统，RAS 进一步
       过滤分布上的重复状态。如果某个状态在分布上已经被充分代表，RAS 会降低
       其传递权重，只保留那些覆盖新奇状态的信号。

    3. 高级皮层：只有同时通过两层过滤的信号才进入高级皮层（如前额叶），
       用于有意识的学习和决策。

    我们的算法模拟这种层级过滤：
    - 首先计算每个片段的时间冗余度 R_temp（预测编码层）
    - 然后计算每个片段的分布冗余度 R_dist（RAS 层）
    - 通过加权融合得到统一冗余度 R_final
    - 选择冗余度最低的 k 个片段作为核心集

    参数 alpha 控制两个过滤层的重要性：
    - alpha = 1.0：仅保留预测编码过滤（纯时间过滤）
    - alpha = 0.0：仅保留 RAS 过滤（纯分布过滤）
    - alpha = 0.75（默认）：时间过滤为主导（符合神经科学中预测编码是主要驱动因素的发现）

    数学定义：
        R_final(e) = alpha * R_temp(e) + (1-alpha) * R_dist(e)
        Coreset = argmin_{S subset E, |S|=k} sum_{e in S} R_final(e)

    对应脑机制：Predictive Coding + RAS 层级过滤

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
        # Convert information utility (score) to redundancy (1 - score)
        # to match the paper's definition: Coreset = argmin sum R_final
        final_scores[ep] = 1.0 - ((alpha * temp_norm[ep]) + (
            (1 - alpha) * dist_norm.get(ep, 0.0)
        ))

    # Sort ascending (minimizing redundancy)
    sorted_eps = sorted(final_scores.items(), key=lambda x: x[1], reverse=False)
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
