import numpy as np
from collections import defaultdict
from src.data_utils import extract_single_arm_action

def compute_temporal_scores(dataset) -> dict[int, float]:
    """
    预测编码启发的时间冗余度评分（Predictive Coding-inspired Temporal Scoring）。

    脑启发机制：
    预测编码理论（Millidge et al., 2021）认为，大脑通过生成关于感官输入的预测来
    理解世界。当实际输入与预测一致时，预测误差很小，神经活动被抑制；只有当输入
    打破预测时，才会产生显著的预测误差，驱动学习和感知更新。

    在机器人演示数据中，连续帧之间的动作变化量 delta_t 可以近似为"预测误差"：
    - 如果 delta_t 很小（机器人静止或匀速运动），说明该帧的动作可以从上一帧预测，
      大脑会将这种可预测的输入"解释掉"，不触发学习信号。
    - 如果 delta_t 很大（动作突变、加速、方向改变），说明实际动作打破了预测，
      产生高预测误差，该帧具有高信息价值。

    因此，我们计算每个片段的平均动作变化量作为其"时间非冗余度"（即信息价值）。
    平均变化量越大的片段，时间冗余度越低，越值得被选中进入核心集。

    数学定义：
        delta_t = ||a_t - a_{t-1}||_2
        S_temp(e) = mean(delta_t) for episode e

    对应脑机制：Predictive Coding（预测编码）
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
