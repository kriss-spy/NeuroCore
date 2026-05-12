---
title: Imitation Learning
category: concepts
tags: [robotics, machine-learning, behavioral-cloning, demonstration-learning]
sources: ["docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).pdf", "docs/papers/OpenVLA: An Open-Source Vision-Language-Action Model.pdf", README.md]
summary: A machine learning paradigm where agents learn policies by imitating expert demonstrations, widely used in robotics for acquiring manipulation skills from human teleoperation.
provenance:
  extracted: 0.72
  inferred: 0.23
  ambiguous: 0.05
base_confidence: 0.96
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Imitation Learning

Imitation learning is a paradigm in which an agent learns to perform tasks by observing and replicating expert demonstrations. In robotics, it is one of the most practical approaches for teaching complex manipulation skills, as it avoids the need for hand-engineered reward functions or extensive environment interaction.

## Key Ideas

- **Behavioral Cloning (BC)**: The simplest form of imitation learning, casting skill acquisition as supervised learning from observation-action pairs. Despite its simplicity, BC remains a strong baseline for many robot learning tasks.
- **Compounding Errors**: A major failure mode of BC is that small prediction errors accumulate over time, causing the agent to drift into out-of-distribution states from which recovery is difficult. This is especially problematic in fine manipulation where millimeter-level errors compound.
- **Action Chunking**: Predicting sequences of actions rather than single timesteps reduces the effective task horizon and mitigates compounding errors. See [[concepts/Action Chunking|Action Chunking]] for details.
- **Generative Modeling**: Human demonstrations are stochastic. Modeling the distribution over action sequences (e.g., via CVAEs) allows policies to capture variability while maintaining precision where required.

## From-Scratch vs. Pretrained Policies

Two major approaches exist for learning visuomotor policies:

1. **From-Scratch Policies**: Train specialized architectures (e.g., transformers, diffusion policies) directly on robot data. Examples include [[entities/ACT|ACT]] (80M parameters) and Diffusion Policy.
2. **Pretrained VLAs**: Fine-tune large vision-language models on robot data, leveraging internet-scale pretraining for generalization. Examples include [[entities/OpenVLA|OpenVLA]] (7B parameters) and RT-2-X (55B parameters).

Pretrained VLAs generally exhibit superior generalization to novel objects, scenes, and language instructions, but are computationally heavier and harder to deploy on resource-constrained hardware.

## Data Quality

The performance of imitation learning depends heavily on the quality and diversity of demonstrations:
- **Temporal Redundancy**: High-frequency recordings contain many low-information frames.
- **Suboptimal Noise**: Human operators may hesitate, correct themselves, or exhibit inconsistent styles.
- **Distribution Bias**: Easy tasks may be overrepresented.

These issues motivate [[concepts/Coreset Selection|coreset selection]] algorithms to extract the most valuable training examples.

## Related

- [[concepts/Action Chunking|Action Chunking]] — a technique to reduce compounding errors in imitation learning
- [[entities/ACT|ACT]] — an imitation learning algorithm for fine manipulation
- [[entities/OpenVLA|OpenVLA]] — a VLA model fine-tuned for robot control
- [[concepts/Vision-Language-Action Model|Vision-Language-Action Model]] — large pretrained models used for imitation learning
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project investigating data selection for imitation learning on robot demonstrations
