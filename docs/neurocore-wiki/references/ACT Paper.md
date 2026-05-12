---
title: "Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)"
category: references
tags: [robotics, imitation-learning, bimanual-manipulation, teleoperation, paper]
sources: ["docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).pdf"]
summary: The original paper introducing ALOHA, a low-cost bimanual teleoperation system, and ACT, an action-chunking imitation learning algorithm for fine manipulation.
provenance:
  extracted: 0.82
  inferred: 0.15
  ambiguous: 0.03
base_confidence: 0.67
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)

**Authors**: Tony Z. Zhao, Vikash Kumar, Sergey Levine, Chelsea Finn  
**Institutions**: Stanford University, UC Berkeley, Meta  
**arXiv**: 2304.13705v1 [cs.RO], April 2023  
**Project**: tonyzhaozh.github.io/aloha

## Summary

This paper presents ALOHA, a low-cost open-source hardware system for bimanual teleoperation (under $20k), and ACT (Action Chunking with Transformers), a novel imitation learning algorithm that learns generative models over action sequences. ACT enables learning of 6 difficult real-world fine manipulation tasks with 80–90% success from only 10 minutes of demonstrations.

## Key Contributions

1. **ALOHA Teleoperation System**: A complete, accessible bimanual teleoperation platform using off-the-shelf ViperX and WidowX robot arms, 3D-printed components, and commodity webcams.
2. **Action Chunking with Transformers (ACT)**: An algorithm that predicts action sequences to reduce effective horizon and combat compounding errors in imitation learning.
3. **Temporal Ensembling**: A technique for smoothing motion by averaging overlapping action predictions at inference time.
4. **CVAE Policy**: Models human demonstration variability using a conditional variational autoencoder, improving robustness to stochastic human behavior.

## ALOHA Hardware

- **Follower Arms**: Two ViperX 6-DoF arms with custom see-through grippers.
- **Leader Arms**: Two WidowX arms for joint-space teleoperation via backdriving.
- **Cameras**: Four Logitech C922x webcams (front, top, two wrist-mounted) at 480×640, 50Hz.
- **Cost**: Approximately $20,000 total, comparable to a single industrial arm.

## ACT Algorithm

- **Action Chunking**: Policy predicts $k$ future target joint positions, reducing effective horizon by $k$-fold.
- **CVAE Training**: Encoder predicts style variable $z$ from observation and action sequence; decoder generates actions conditioned on $z$ and observation.
- **Loss**: Reconstruction loss (L1) + $\beta$-weighted KL divergence regularization.
- **Inference**: Query policy every timestep; ensemble overlapping predictions with exponential weights $w_i = \exp(-m \cdot i)$.

## Experimental Results

- **Real-world tasks**: Threading velcro ties, preparing tape, putting on shoes, opening cups, slotting batteries.
- **Simulated tasks**: Transfer Cube, Open Cup (from the ALOHA Sim dataset used in [[projects/NeuroCore/NeuroCore|NeuroCore]]).
- **Data efficiency**: 80–90% success with only ~50 demonstrations (10 minutes of teleoperation).

## Key Takeaways

- Low-cost hardware can perform high-precision manipulation when paired with strong imitation learning algorithms and closed-loop visual feedback.
- Action chunking and temporal ensembling are simple yet effective techniques for improving imitation learning on high-frequency, precision-demanding tasks.
- Joint-space mapping is preferable to task-space mapping for fine manipulation near kinematic singularities.

## Related

- [[entities/ACT|ACT]] — the algorithm and system entity page
- [[concepts/Action Chunking|Action Chunking]] — the core technique
- [[references/OpenVLA Paper|OpenVLA Paper]] — a VLA model that builds on insights from ACT and ALOHA
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project using the ALOHA Sim Transfer Cube dataset
