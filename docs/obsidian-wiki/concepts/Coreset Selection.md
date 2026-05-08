---
title: Coreset Selection
category: concepts
tags: [data-pruning, coreset, neural-scaling-laws, dataset-efficiency]
sources: ["docs/papers/Beyond neural scaling laws: beating power law scaling via data pruning.pdf", "docs/papers/Predictive Coding: a Theoretical and Experimental Review.pdf", README.md]
summary: The practice of selecting a small, high-information subset of training data that preserves or improves model performance compared to training on the full dataset.
provenance:
  extracted: 0.68
  inferred: 0.27
  ambiguous: 0.05
base_confidence: 0.96
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Coreset Selection

Coreset selection, also referred to as data pruning, is the problem of identifying and retaining only the most informative subset of a training dataset. The goal is to train models on a smaller pruned dataset without sacrificing—and sometimes even improving—downstream performance. This is particularly relevant for robotics, where human demonstration datasets often contain significant temporal redundancy, distributional imbalance, and suboptimal noise.

## Key Ideas

- Traditional neural scaling laws suggest that test error falls off as a power law with dataset size, implying that simply collecting more data yields diminishing returns.
- Data pruning can break power law scaling: by selecting high-quality subsets, error can decrease exponentially with the size of the pruned dataset rather than polynomially.
- The optimal pruning strategy depends on data abundance. When data is abundant, retaining "hard" examples (those the model finds most difficult) is optimal; when data is scarce, retaining "easy" examples performs better. ^[inferred]
- The quality of the pruning metric is the critical determinant of pruning success. Poor metrics can lead to worse-than-random performance.

## Pruning Metrics

A variety of metrics have been proposed to quantify the importance or difficulty of individual training examples:

- **EL2N (Error L2 Norm)**: Measures the average L2 norm of the error vector for each example, typically computed using a small ensemble trained for a short duration.
- **Forgetting Score**: Counts how many times an example transitions from correctly classified to incorrectly classified during training.
- **Memorization Score**: Quantifies how much the probability of correctly predicting an example increases when it is included in the training set.
- **Self-Supervised Prototypicality**: Uses k-means clustering in the embedding space of a self-supervised model; examples far from their cluster centroid are considered "hard" and retained.

## Brain-Inspired Perspectives

The human brain provides a natural model for coreset selection:

- **Predictive Coding**: The brain does not passively process all sensory frames; it generates predictions and only allocates significant neural resources to stimuli that violate those predictions (large prediction errors). This is functionally equivalent to filtering out temporally redundant "easy" examples. ^[inferred]
- **Reticular Activating System (RAS)**: The brain's RAS filters background noise and directs attention toward high-utility sensory events based on task goals. This aligns with the idea of pruning data by estimated information gain or task relevance. ^[inferred]

## Practical Considerations

- **Class Balancing**: Pruning metrics often amplify class imbalance. Maintaining a balanced class distribution in the pruned subset is essential for preserving performance.
- **Compute Trade-offs**: While pruning reduces per-epoch training cost, aggressively pruned datasets may require more epochs to converge. The net compute savings depend on the specific metric and dataset.
- **Scalability**: Many supervised pruning metrics do not scale well to large datasets like ImageNet, motivating the development of simple, self-supervised alternatives.

## Related Concepts

- [[concepts/Neural Scaling Laws|Neural Scaling Laws]] — the empirical relationship between dataset size, model size, and performance
- [[concepts/Predictive Coding|Predictive Coding]] — a brain theory that naturally implements temporal redundancy filtering
- [[references/Data Pruning Paper|Beyond neural scaling laws: beating power law scaling via data pruning]] — the foundational paper on data pruning theory and metrics
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project applying coreset selection to robot demonstration data
