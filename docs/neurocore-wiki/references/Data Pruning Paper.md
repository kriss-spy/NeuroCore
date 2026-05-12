---
title: "Beyond neural scaling laws: beating power law scaling via data pruning"
category: references
tags: [data-pruning, scaling-laws, coreset-selection, paper]
sources: ["docs/papers/Beyond neural scaling laws: beating power law scaling via data pruning.pdf"]
summary: A NeurIPS 2022 paper showing that high-quality data pruning metrics can break power law scaling and achieve exponential scaling of error with pruned dataset size.
provenance:
  extracted: 0.80
  inferred: 0.17
  ambiguous: 0.03
base_confidence: 0.67
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Beyond neural scaling laws: beating power law scaling via data pruning

**Authors**: Ben Sorscher, Robert Geirhos, Shashank Shekhar, Ari S. Morcos, Surya Ganguli  
**Institutions**: Stanford University, University of Tübingen, Meta AI (FAIR)  
**Conference**: NeurIPS 2022  
**arXiv**: 2206.14486v6 [cs.LG], April 2023

## Summary

This paper demonstrates both theoretically and empirically that data pruning can break the power law scaling of test error with dataset size, potentially achieving exponential scaling instead. The authors develop an analytic theory of data pruning in the student-teacher perceptron setting, benchmark 10 pruning metrics on ImageNet, and introduce a simple self-supervised prototypicality metric that matches the best supervised metrics.

## Key Contributions

1. **Analytic Theory**: Using statistical mechanics, the authors derive predictions for data pruning in a teacher-student perceptron model:
   - The optimal pruning strategy depends on data abundance: retain hard examples when data is abundant, easy examples when data is scarce.
   - Exponential scaling of test error with pruned dataset size is possible under Pareto-optimal pruning.
2. **Empirical Validation**: Demonstrates signatures of exponential scaling on ResNets trained on CIFAR-10, SVHN, and ImageNet, as well as Vision Transformers fine-tuned on CIFAR-10.
3. **Large-Scale Benchmarking**: Evaluates 10 pruning metrics on ImageNet, finding that most metrics developed for small datasets scale poorly.
4. **Self-Supervised Prototypicality Metric**: Uses k-means clustering in a self-supervised embedding space (SWaV) to rank examples by prototypicality, achieving comparable performance to the best supervised metric (memorization) without requiring labels.

## Pruning Metrics Benchmarked

| Metric | Supervised | Scales to ImageNet | Notes |
|---|---|---|---|
| EL2N | Yes | Partially | Requires ensemble training |
| Forgetting Score | Yes | No | Tracks classification transitions |
| Memorization | Yes | Yes | Most compute-intensive; best performance |
| Ensemble Active Learning | Yes | No | Expensive ensemble training |
| DDD (Diverse Ensembles) | Yes | Unexplored | Counts misclassifications across models |
| Self-Supervised Prototypes | No | Yes | Best unsupervised metric; simple and scalable |

## Key Findings

- **Class Balancing**: Pruning metrics tend to amplify class imbalance; enforcing a 50% class-balancing ratio is essential for maintaining performance.
- **Metric Quality is Critical**: The difference between good and bad pruning metrics is larger than the difference between pruning and random selection.
- **Foundation Datasets**: Carefully pruned subsets of large unlabeled datasets could serve as "foundation datasets," amortizing pruning costs across many downstream training runs.

## Limitations

- Achieving exponential scaling requires a high-quality pruning metric, and most existing metrics do not scale well to large datasets.
- Training on pruned datasets for the same number of iterations as the full dataset (more epochs) sometimes improves performance but partially offsets computational savings.
- Fairness implications of pruning across demographic groups require careful analysis in deployment settings.

## Related

- [[concepts/Coreset Selection|Coreset Selection]] — the concept page on data pruning and coreset selection
- [[concepts/Neural Scaling Laws|Neural Scaling Laws]] — the empirical relationship between data, model size, and performance
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project applying data pruning to robot demonstration datasets
