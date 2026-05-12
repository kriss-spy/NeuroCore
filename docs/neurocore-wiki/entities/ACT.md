---
title: ACT
category: entities
tags: [imitation-learning, robotics, bimanual-manipulation, teleoperation, action-chunking]
sources: ["docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT).pdf", README.md]
summary: Action Chunking with Transformers (ACT) is an imitation learning algorithm and low-cost teleoperation system for fine-grained bimanual manipulation.
provenance:
  extracted: 0.78
  inferred: 0.17
  ambiguous: 0.05
base_confidence: 0.77
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# ACT

Action Chunking with Transformers (ACT) is an imitation learning algorithm developed by Tony Z. Zhao et al. for learning fine-grained bimanual manipulation skills. It is paired with **ALOHA** (A Low-cost Open-source Hardware System for Bimanual Teleoperation), a complete teleoperation and data collection platform built for under $20,000.

## ALOHA Hardware System

ALOHA enables precise, contact-rich, and dynamic bimanual teleoperation through:
- **Two ViperX 6-DoF robot arms** (follower) with custom 3D-printed see-through fingers.
- **Two WidowX arms** (leader) for joint-space teleoperation via backdriving.
- **Four Logitech C922x webcams** (front, top, and two wrist-mounted) streaming 480×640 RGB images at 50Hz.
- A 3D-printed "handle and scissor" mechanism and rubber band load balancing to reduce operator fatigue.

The system can perform tasks such as threading zip ties, inserting RAM into motherboards, opening condiment cups, and juggling ping pong balls.

## ACT Algorithm

ACT addresses two key challenges in imitation learning for fine manipulation:

### 1. Compounding Errors

Small errors in predicted actions accumulate over time, causing the robot to drift outside its training distribution. ACT mitigates this through **action chunking**: the policy predicts a sequence of $k$ future target joint positions rather than a single action, reducing the effective task horizon by a factor of $k$.

### 2. Human Demonstration Variability

Human operators exhibit variability in how they execute tasks. ACT models this by training the policy as a **Conditional Variational Autoencoder (CVAE)**:
- A CVAE encoder predicts the distribution of a style variable $z$ given the observation and action sequence.
- A CVAE decoder (the policy) generates action sequences conditioned on $z$ and the current observation.
- At inference, $z$ is set to zero for deterministic decoding.

### 3. Temporal Ensembling

To ensure smooth motion, the policy is queried at every timestep. Overlapping action chunks are combined via an exponential weighted average, with weights $w_i = \exp(-m \cdot i)$ where $i$ is the prediction horizon.

## Implementation Details

- **Architecture**: ResNet18 image encoders, transformer encoder (cross-modal fusion), transformer decoder (sequence generation).
- **Observation Space**: 4 RGB images + 14-DoF joint positions (7 per arm).
- **Action Space**: Absolute target joint positions for both arms ($k \times 14$ tensor).
- **Model Size**: ~80M parameters.
- **Training**: ~5 hours on a single RTX 2080 Ti GPU.

## Performance

ACT achieves 80–90% success on 6 real-world fine manipulation tasks with only 10 minutes (approximately 50 demonstrations) of human data, significantly outperforming prior methods such as Behavioral Cloning and Behavior Transformers (BeT).

## Related

- [[concepts/Action Chunking|Action Chunking]] — the core technique underlying ACT
- [[concepts/Imitation Learning|Imitation Learning]] — the learning paradigm ACT employs
- [[references/ACT Paper|Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)]] — the original paper
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project using the ALOHA Sim dataset for coreset selection experiments
