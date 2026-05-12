---
title: "OpenVLA: An Open-Source Vision-Language-Action Model"
category: references
tags: [vla, robotics, open-source, paper]
sources: ["docs/papers/OpenVLA: An Open-Source Vision-Language-Action Model.pdf"]
summary: The original paper introducing OpenVLA, a 7B-parameter open-source VLA trained on 970k robot episodes that outperforms RT-2-X.
provenance:
  extracted: 0.85
  inferred: 0.12
  ambiguous: 0.03
base_confidence: 0.67
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# OpenVLA: An Open-Source Vision-Language-Action Model

**Authors**: Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti, et al.  
**Institutions**: Stanford, UC Berkeley, Toyota Research Institute, Google DeepMind, Physical Intelligence, MIT  
**arXiv**: 2406.09246v3 [cs.RO], September 2024  
**Project**: https://openvla.github.io

## Summary

This paper introduces OpenVLA, a 7B-parameter open-source vision-language-action model for generalist robot manipulation. OpenVLA is trained on 970k real-world robot demonstrations from the Open X-Embodiment dataset and outperforms the 55B-parameter closed-source RT-2-X by 16.5% absolute success rate across 29 tasks, despite having 7× fewer parameters.

## Key Contributions

1. **Open-source generalist VLA**: First large-scale open-source VLA with publicly available weights, training code, and fine-tuning notebooks.
2. **Strong zero-shot performance**: Outperforms RT-2-X on BridgeData V2 and matches it on Google Robot evaluations.
3. **Efficient fine-tuning**: Demonstrates effective adaptation to new robot setups via full fine-tuning, LoRA, and quantization.
4. **Architecture insights**: Shows that fusing DINOv2 and SigLIP visual features improves spatial reasoning and generalization for robot control.

## Model Architecture

- **Backbone**: Prismatic-7B VLM (SigLIP + DINOv2 vision encoders, Llama 2 7B language model, 2-layer MLP projector).
- **Action Representation**: Each action dimension discretized into 256 bins using quantile-based boundaries; mapped into Llama tokenizer vocabulary.
- **Training Objective**: Next-token prediction with cross-entropy loss evaluated only on action tokens.

## Experimental Results

- **Multi-robot generalization**: Evaluated on WidowX (BridgeData V2) and Google Robot platforms.
- **Fine-tuning**: Tested on Franka-Tabletop and Franka-DROID setups with 10–150 demonstrations.
- **Efficiency**: LoRA fine-tuning and 4-bit/8-bit quantization enable deployment on consumer GPUs without significant accuracy loss.

## Key Takeaways

- Open-source VLAs can match or exceed closed-source counterparts when trained on sufficiently large and diverse datasets.
- Data curation (e.g., filtering all-zero actions) and dataset size matter substantially.
- Parameter-efficient fine-tuning methods developed for language models transfer effectively to VLAs.

## Related

- [[entities/OpenVLA|OpenVLA]] — the model entity page
- [[concepts/Vision-Language-Action Model|Vision-Language-Action Model]] — the model paradigm
- [[references/ACT Paper|Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (ACT)]] — related work on robot manipulation
