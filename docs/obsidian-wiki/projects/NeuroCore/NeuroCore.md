---
title: NeuroCore
category: project
tags: [vla, coreset-selection, predictive-coding, embodied-ai, robotics]
sources: [README.md, "docs/papers/OpenVLA: An Open-Source Vision-Language-Action Model.pdf", "docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).pdf", "docs/papers/Beyond neural scaling laws: beating power law scaling via data pruning.pdf", "docs/papers/Predictive Coding: a Theoretical and Experimental Review.pdf"]
summary: A course project exploring brain-inspired coreset selection for lightweight VLA robotic arm action prediction on the ALOHA simulation dataset.
provenance:
  extracted: 0.65
  inferred: 0.30
  ambiguous: 0.05
base_confidence: 0.98
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# NeuroCore

NeuroCore is a course project that investigates how brain-inspired data selection mechanisms can improve the efficiency of training lightweight Vision-Language-Action (VLA) models for robotic manipulation. The project is motivated by the observation that real-world robot demonstration datasets contain massive temporal and distributional redundancy, making full-dataset training computationally wasteful. By drawing inspiration from the human brain's predictive coding and reticular activating system (RAS), the project designs automated data-pruning algorithms to extract high-value coresets from the ALOHA Sim Transfer Cube dataset.

## Key Concepts

- [[concepts/Vision-Language-Action Model|Vision-Language-Action Model]] — the model paradigm that maps visual observations and language instructions directly to robot control actions
- [[concepts/Coreset Selection|Coreset Selection]] — selecting a small, high-quality subset of training data that preserves or improves model performance
- [[concepts/Predictive Coding|Predictive Coding]] — a brain-inspired mechanism for filtering temporal redundancy by only attending to prediction errors
- [[concepts/Action Chunking|Action Chunking]] — predicting sequences of actions to reduce compounding errors in imitation learning
- [[concepts/Embodied AI|Embodied AI]] — the broader field of integrating AI with physical robot embodiments

## Research Goals

1. **Baseline**: Randomly sample 10% of trajectories from the ALOHA Sim dataset, extract visual features with frozen ResNet-18 or CLIP, and train a lightweight MLP to predict 7-DoF arm actions. Report MSE.
2. **Brain-Inspired Coreset Selection**: Design an automated pruning algorithm (e.g., based on continuous action variance or visual feature clustering) to intelligently select the most valuable 10% coreset.
3. **Validation**: Retrain the MLP on the selected 10% coreset and compare MSE against the random baseline to demonstrate that higher-quality data subsets yield better model performance.

## Dataset

The project uses the **ALOHA Sim Transfer Cube (Human Demonstrations)** dataset from Hugging Face LeRobot, containing 50 successful human demonstration episodes with multi-view camera images and 14-DoF joint action labels.

## Related Work

- [[entities/OpenVLA|OpenVLA]] — a 7B-parameter open-source VLA trained on 970k robot episodes
- [[entities/ACT|ACT]] — the Action Chunking with Transformers algorithm and ALOHA teleoperation system
- [[references/Data Pruning Paper|Beyond neural scaling laws: beating power law scaling via data pruning]] — theoretical and empirical foundations for data pruning

## Open Questions

- How should "redundancy" be formally defined for multi-modal robot demonstration data?
- Can predictive coding's precision-weighting be operationalized as a trainable attention mechanism for data selection?
- What is the optimal trade-off between pruning aggressiveness and downstream task performance for small-scale (50-episode) datasets?
