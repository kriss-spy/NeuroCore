"""
Validate the coreset selection by retraining the MLP on the selected episodes
and comparing against the random baseline.
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from src.baseline import run_baseline


def run_validation(
    coreset_path: str = "results/coreset_selection.json",
    feature_path: str = "results/features_resnet18.pt",
    baseline_metrics_path: str = "results/baseline_metrics.json",
    save_dir: str = "results/coreset",
) -> dict:
    """
    Retrain the MLP on the coreset episodes and compare with the random baseline.

    Returns a dict with coreset_metrics, baseline_metrics, and comparison.
    """
    with open(coreset_path, "r", encoding="utf-8") as f:
        coreset_data = json.load(f)
    selected_episodes = coreset_data["selected_episodes"]

    print(f"Training on coreset episodes: {selected_episodes}")
    coreset_metrics = run_baseline(
        feature_path=feature_path,
        train_episodes=selected_episodes,
        save_dir=save_dir,
    )

    with open(baseline_metrics_path, "r", encoding="utf-8") as f:
        baseline_metrics = json.load(f)

    print("\n=== Results ===")
    print(f"Random Baseline MSE: {baseline_metrics['mse']:.6f}")
    print(f"Coreset MSE:         {coreset_metrics['mse']:.6f}")
    improvement = baseline_metrics["mse"] - coreset_metrics["mse"]
    pct = (improvement / baseline_metrics["mse"]) * 100
    print(f"Improvement:         {improvement:.6f} ({pct:+.2f}%)")

    # Save comparison
    os.makedirs(save_dir, exist_ok=True)
    comparison = {
        "baseline_mse": baseline_metrics["mse"],
        "coreset_mse": coreset_metrics["mse"],
        "improvement": improvement,
        "improvement_percent": pct,
        "selected_episodes": selected_episodes,
    }
    with open(os.path.join(save_dir, "coreset_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    # Generate comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(baseline_metrics["val_losses"], label="Random Baseline", color="gray", linestyle="--")
    ax1.plot(coreset_metrics["val_losses"], label="Coreset", color="blue")
    ax1.set_title("Validation Loss Convergence")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MSE")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    joints = list(range(7))
    width = 0.35
    ax2.bar(
        [j - width / 2 for j in joints],
        baseline_metrics["per_joint_mse"],
        width,
        label="Random Baseline",
        color="gray",
    )
    ax2.bar(
        [j + width / 2 for j in joints],
        coreset_metrics["per_joint_mse"],
        width,
        label="Coreset",
        color="blue",
    )
    ax2.set_title("Per-Joint Test MSE")
    ax2.set_xlabel("Joint Index")
    ax2.set_ylabel("MSE")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(fig_path, dpi=150)
    print(f"Saved comparison figure to {fig_path}")

    return {
        "coreset_metrics": coreset_metrics,
        "baseline_metrics": baseline_metrics,
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_validation()