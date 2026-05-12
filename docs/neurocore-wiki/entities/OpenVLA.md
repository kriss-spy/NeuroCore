---
title: OpenVLA
category: entities
tags: [vla, robotics, open-source, llm, multimodal]
sources: ["docs/papers/OpenVLA: An Open-Source Vision-Language-Action Model.pdf"]
summary: A 7B-parameter open-source vision-language-action model trained on 970k robot demonstrations, outperforming closed-source RT-2-X on multiple benchmarks.
provenance:
  extracted: 0.80
  inferred: 0.15
  ambiguous: 0.05
base_confidence: 0.67
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# OpenVLA

OpenVLA is a 7B-parameter open-source Vision-Language-Action (VLA) model developed by researchers at Stanford, UC Berkeley, Toyota Research Institute, and other institutions. It is the first large-scale generalist VLA to be fully open-sourced, including model checkpoints, training code, and fine-tuning notebooks.

## Architecture

OpenVLA builds on the **Prismatic-7B** VLM backbone, which consists of:
- **Vision Encoder**: A fused encoder combining DINOv2 (for spatial reasoning) and SigLIP (for semantic understanding), outputting concatenated patch features.
- **Projector**: A 2-layer MLP that maps visual features into the language model embedding space.
- **Language Model Backbone**: Llama 2 7B, which processes fused visual and language tokens and generates robot action tokens.

## Action Tokenization

Robot actions are represented in the LLM's output space by:
- Discretizing each action dimension into 256 bins, with bin boundaries set by the 1st and 99th quantiles of the training data (ignoring outliers).
- Overwriting the 256 least-used tokens in the Llama tokenizer vocabulary with these action tokens.
- Training with standard next-token prediction, evaluating cross-entropy loss only on action tokens.

## Training Data

OpenVLA is trained on **970k real-world robot manipulation trajectories** from the Open X-Embodiment dataset, spanning a wide diversity of robot embodiments, tasks, and scenes.

## Performance Highlights

- Outperforms the 55B-parameter closed-source **RT-2-X** by **16.5% absolute success rate** across 29 evaluation tasks on WidowX and Google Robot embodiments.
- Achieves strong generalization across visual, motion, physical, and semantic axes.
- Demonstrates robust language grounding in multi-object scenes.

## Fine-Tuning and Efficiency

OpenVLA supports efficient adaptation to new robot setups:
- **Full Fine-Tuning**: Effective with as few as 10–150 demonstrations.
- **LoRA (Low-Rank Adaptation)**: Enables parameter-efficient fine-tuning on consumer GPUs.
- **Quantization**: 8-bit and 4-bit quantized inference runs efficiently without significant performance degradation.

## Limitations

- Currently supports only single-image observations; multi-image, proprioceptive, and historical observations are not yet integrated.
- Inference throughput is insufficient for high-frequency control setups such as ALOHA (50Hz).
- Typical task success rates remain below 90%, indicating room for reliability improvements.

## Related

- [[concepts/Vision-Language-Action Model|Vision-Language-Action Model]] — the model paradigm OpenVLA implements
- [[references/OpenVLA Paper|OpenVLA: An Open-Source Vision-Language-Action Model]] — the original paper
- [[entities/ACT|ACT]] — a from-scratch imitation learning method that OpenVLA outperforms when fine-tuned
