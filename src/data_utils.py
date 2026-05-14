from datasets import load_dataset


# Cache the dataset so repeated calls are cheap
_dataset_cache = None

# NOTE: Language Instruction Investigation
# The ALOHA Sim Transfer Cube (Human Demonstrations) dataset does NOT
# contain language instruction annotations. After inspecting all fields
# (observation.state, action, episode_index, frame_index, timestamp,
# next.done, index, task_index), there is no "language", "instruction",
# or "text" field. The task_index is uniformly 0 across all 50 episodes,
# indicating a single-task dataset without natural language labels.
# Therefore, per the course guidance, we proceed with visual-only
# features for this lightweight validation experiment.


def load_aloha_dataset(split: str = "train"):
    """
    Loads the aloha_sim_transfer_cube_human dataset from HuggingFace.
    Returns the dataset object (actions, episode_index, frame_index, etc.).
    Images are NOT included in this stream; visual features should be
    loaded from the cached ResNet-18 extraction.
    """
    global _dataset_cache
    if _dataset_cache is None:
        ds = load_dataset("lerobot/aloha_sim_transfer_cube_human")
        _dataset_cache = ds[split]
    return _dataset_cache


def extract_single_arm_action(action):
    """
    Isolate the single-arm 7-DoF action from the 14-DoF action tensor.
    We take the first 7 dimensions.
    """
    if len(action) >= 7:
        return action[:7]
    return action
