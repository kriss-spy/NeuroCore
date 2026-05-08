import torch
from lerobot.datasets import LeRobotDataset


def load_aloha_dataset():
    """
    Loads the aloha_sim_transfer_cube_human dataset from HuggingFace using LeRobot.
    Returns the dataset object.
    """
    # Load dataset
    dataset = LeRobotDataset("lerobot/aloha_sim_transfer_cube_human")
    return dataset


def extract_single_arm_action(action_tensor):
    """
    Isolate the single-arm 7-DoF action from the 14-DoF action tensor.
    We take the first 7 dimensions.
    """
    if len(action_tensor) >= 7:
        return action_tensor[:7]
    return action_tensor
