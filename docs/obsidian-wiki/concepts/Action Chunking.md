---
title: Action Chunking
category: concepts
tags: [imitation-learning, robotics, action-prediction, temporal-modeling]
sources: ["docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).pdf", README.md]
summary: A neuroscience-inspired technique where policies predict sequences of future actions rather than single timesteps, reducing effective horizon and compounding errors.
provenance:
  extracted: 0.75
  inferred: 0.20
  ambiguous: 0.05
base_confidence: 0.77
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Action Chunking

Action chunking is a technique in imitation learning where the policy predicts a sequence of future actions (a "chunk") rather than a single action at each timestep. Inspired by the neuroscience concept that individual actions are grouped and executed as unified units, action chunking reduces the effective horizon of long trajectories and mitigates the compounding error problem inherent in behavioral cloning.

## Key Ideas

- Instead of modeling $\pi_\theta(a_t | s_t)$, the policy models $\pi_\theta(a_{t:t+k} | s_t)$, predicting the next $k$ actions conditioned on the current observation.
- This reduces the effective task horizon by a factor of $k$, making long-horizon tasks more tractable for imitation learning.
- Action chunking also helps model **non-Markovian behavior** in human demonstrations, such as pauses or temporally correlated confounders that single-step policies struggle to capture.

## Temporal Ensembling

A naive implementation of action chunking would only query the policy every $k$ steps, leading to jerky motion. **Temporal ensembling** addresses this by:
- Querying the policy at every timestep, causing overlapping action chunks.
- Combining multiple predictions for the same timestep using a weighted average with exponential weighting $w_i = \exp(-m \cdot i)$, where $i$ indexes how far into the future a prediction was made.
- This produces smoother trajectories without additional training cost, only increased inference computation.

## Modeling Human Variability

Human demonstrations are inherently stochastic: given the same observation, different operators may choose different trajectories. To capture this variability while maintaining precision where it matters:
- The policy can be trained as a **Conditional Variational Autoencoder (CVAE)**, learning a generative model over action sequences.
- At test time, the latent variable $z$ is set to the mean of the prior (typically zero) for deterministic decoding.
- The CVAE objective combines reconstruction loss with a KL divergence regularization term weighted by hyperparameter $\beta$.

## Relation to VLA Models

While action chunking was originally developed for smaller from-scratch policies like [[entities/ACT|ACT]], it is directly applicable to VLA models. Large VLAs such as [[entities/OpenVLA|OpenVLA]] currently predict single actions and suffer from low inference throughput; integrating action chunking or speculative decoding is a promising direction for enabling high-frequency control. ^[inferred]

## Related Concepts

- [[entities/ACT|ACT]] — the Action Chunking with Transformers algorithm and ALOHA system
- [[concepts/Imitation Learning|Imitation Learning]] — the learning paradigm in which action chunking is applied
- [[concepts/Vision-Language-Action Model|Vision-Language-Action Model]] — large multimodal policies that could benefit from action chunking
- [[references/ACT Paper|Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)]] — the original paper introducing action chunking for robotics
