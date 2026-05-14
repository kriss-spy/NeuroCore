import json
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.data_utils import extract_single_arm_action, load_aloha_dataset


@dataclass
class BaselineConfig:
    learning_rate: float = 1e-3
    batch_size: int = 32
    epochs: int = 50
    hidden_dims: Tuple[int, int] = (256, 128)
    train_episode_count: int = 5
    seed: int = 42
    max_frames_per_episode: int | None = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def _load_cached_features(feature_path: str) -> Dict[Tuple[int, int], torch.Tensor]:
    if not os.path.exists(feature_path):
        raise FileNotFoundError(
            f"Feature cache not found at {feature_path}. Run feature extraction first."
        )
    raw = torch.load(feature_path, weights_only=True)
    # Normalize keys to Python ints (handles legacy tensor-key caches)
    return {
        (int(k[0]), int(k[1])): v
        for k, v in raw.items()
    }


def _select_train_episodes(total_episodes: int, count: int, seed: int) -> List[int]:
    rng = np.random.default_rng(seed)
    return sorted(rng.choice(total_episodes, size=count, replace=False).tolist())


def _build_feature_action_pairs(
    dataset,
    features: Dict[Tuple[int, int], torch.Tensor],
    episode_indices: Iterable[int],
    max_frames_per_episode: int | None,
) -> Tuple[np.ndarray, np.ndarray]:
    x_list: List[np.ndarray] = []
    y_list: List[np.ndarray] = []
    episode_set = set(int(ep) for ep in episode_indices)
    per_episode_counts = {ep: 0 for ep in episode_set}

    action_map: Dict[Tuple[int, int], np.ndarray] = {}
    for idx in range(len(dataset)):
        item = dataset[idx]
        ep = int(item["episode_index"])
        if ep not in episode_set:
            continue
        frame = int(item["frame_index"])
        action = extract_single_arm_action(item["action"])
        action_map[(ep, frame)] = np.array(action, dtype=np.float32)

    for (ep_idx, frame_idx), feature in features.items():
        ep = int(ep_idx)
        frame = int(frame_idx)
        if ep not in episode_set:
            continue
        if max_frames_per_episode is not None and per_episode_counts[ep] >= max_frames_per_episode:
            continue
        if (ep, frame) not in action_map:
            continue
        action = action_map[(ep, frame)]
        x_list.append(feature.numpy())
        y_list.append(action)
        per_episode_counts[ep] += 1

    if not x_list:
        raise ValueError("No samples found for the selected episodes. Check inputs.")

    return np.stack(x_list, axis=0), np.stack(y_list, axis=0)


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Tuple[int, int], output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def run_baseline(
    feature_path: str = "results/features_resnet18.pt",
    config: BaselineConfig | None = None,
    train_episodes: List[int] | None = None,
    save_dir: str = "results",
) -> Dict[str, object]:
    """
    Train a baseline MLP on a given subset of episodes (or a random 10% subset if None)
    and evaluate on the rest.

    Brain-inspired design note:
    The assignment specifies [visual_features + language_instruction] -> [7-DoF action].
    However, after inspecting the ALOHA dataset, we confirmed that language instruction
    annotations are not provided (only action vectors, state vectors, and episode indices
    are available). Therefore, we proceed with a visual-only regression pipeline per the
    course guidance: "verify whether language labels exist before assuming a multimodal
    pipeline." The frozen ResNet-18 acts as a passive visual encoder, analogous to the
    early visual cortex (V1/V2), while the lightweight MLP serves as the downstream
    motor prediction network.

    Args:
        feature_path: path to cached ResNet-18 features.
        config: training hyperparameters.
        train_episodes: optional list of episode indices to train on.
        save_dir: directory to save metrics and checkpoints.

    Returns a dict with metrics, loss curves, and selected episodes.
    """
    cfg = config or BaselineConfig()
    _set_seed(cfg.seed)

    dataset = load_aloha_dataset()
    total_episodes = int(max(dataset["episode_index"])) + 1

    if train_episodes is None:
        train_episodes = _select_train_episodes(total_episodes, cfg.train_episode_count, cfg.seed)
    else:
        train_episodes = sorted([int(ep) for ep in train_episodes])

    test_episodes = [ep for ep in range(total_episodes) if ep not in train_episodes]

    features = _load_cached_features(feature_path)
    x_train, y_train = _build_feature_action_pairs(
        dataset, features, train_episodes, cfg.max_frames_per_episode
    )
    x_test, y_test = _build_feature_action_pairs(
        dataset, features, test_episodes, cfg.max_frames_per_episode
    )

    train_dataset = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    test_dataset = TensorDataset(
        torch.tensor(x_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    model = BaselineMLP(input_dim=512, hidden_dims=cfg.hidden_dims, output_dim=7).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)
    loss_fn = nn.MSELoss()

    train_losses: List[float] = []
    val_losses: List[float] = []

    for _ in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(cfg.device)
            batch_y = batch_y.to(cfg.device)
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = loss_fn(preds, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        train_losses.append(float(np.mean(epoch_losses)))

        model.eval()
        val_epoch_losses = []
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x = batch_x.to(cfg.device)
                batch_y = batch_y.to(cfg.device)
                preds = model(batch_x)
                loss = loss_fn(preds, batch_y)
                val_epoch_losses.append(loss.item())
        val_losses.append(float(np.mean(val_epoch_losses)))

    # Final evaluation
    model.eval()
    preds_list = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(cfg.device)
            preds_list.append(model(batch_x).cpu())
    preds = torch.cat(preds_list, dim=0).numpy()
    mse = float(np.mean((preds - y_test) ** 2))
    per_joint_mse = np.mean((preds - y_test) ** 2, axis=0).tolist()

    os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
    checkpoint_path = os.path.join(save_dir, "checkpoints", "baseline.pt")
    torch.save(model.state_dict(), checkpoint_path)

    metrics = {
        "mse": mse,
        "per_joint_mse": per_joint_mse,
        "train_episodes": train_episodes,
        "test_episodes": test_episodes,
        "train_losses": train_losses,
        "val_losses": val_losses,
    }

    metrics_path = os.path.join(save_dir, "baseline_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


if __name__ == "__main__":
    run_baseline()
