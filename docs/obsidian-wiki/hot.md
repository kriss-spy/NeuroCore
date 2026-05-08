---
title: Hot Cache
updated: 2026-05-08T13:30:00+08:00
---

# Hot Cache

*A ~500-word semantic snapshot of recent activity. Updated after every major write operation.*

## Recent Activity

- [2026-05-08T13:30:00+08:00] INGEST — First major ingest: distilled README.md and 4 foundational papers into 14 wiki pages covering brain-inspired coreset selection for VLA robotics.

## Active Threads

1. **Brain-Inspired Data Selection for Robotics**: The NeuroCore project is exploring how predictive coding and RAS-like attention mechanisms can filter redundant robot demonstration data. Open questions include how to operationalize prediction-error-based pruning for small (50-episode) datasets and whether precision-weighting can be learned as a data-selection attention mechanism.
2. **VLA Efficiency and Deployment**: OpenVLA demonstrates that 7B-parameter open-source VLAs can outperform 55B closed-source models, but inference throughput remains a bottleneck for high-frequency control. Action chunking and quantization are promising directions for bridging this gap.
3. **Scaling Laws vs. Data Pruning**: The data pruning literature suggests exponential scaling is possible with high-quality metrics, but most metrics fail to scale to ImageNet. Self-supervised prototypicality offers a compelling path forward for unlabeled robotics data.

## Key Takeaways

- **Predictive coding provides a principled framework for temporal redundancy filtering**: The brain's prediction-error minimization naturally ignores predictable frames—directly applicable to filtering "staring frames" in robot demonstrations.
- **Data pruning can beat power law scaling, but metric quality is the bottleneck**: Only a few pruning metrics scale to large datasets; self-supervised embedding-space methods are the most promising for robotics where labels are expensive.
- **Open-source VLAs have reached competitive performance**: OpenVLA outperforms RT-2-X despite 7× fewer parameters, validating the approach of fine-tuning internet-pretrained VLMs for robot control.

## Flagged Contradictions

*None yet.*
