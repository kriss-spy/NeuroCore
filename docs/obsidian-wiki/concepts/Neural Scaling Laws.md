---
title: Neural Scaling Laws
category: concepts
tags: [scaling-laws, deep-learning, data-efficiency, theory]
sources: ["docs/papers/Beyond neural scaling laws: beating power law scaling via data pruning.pdf"]
summary: Empirical observations that test error typically falls off as a power law with training set size, model size, or compute, motivating research into more efficient scaling strategies like data pruning.
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

# Neural Scaling Laws

Neural scaling laws are empirically observed relationships in deep learning where test error typically decreases as a power law with respect to training set size, model parameter count, or computational budget. These laws have motivated massive investments in data collection and compute, but their shallow exponents suggest potentially inefficient use of resources.

## Key Ideas

- Test loss $L$ often scales as $L \approx P^{-\nu}$ where $P$ is the number of training examples and $\nu$ is a small exponent (e.g., $\nu \approx 0.095$ for large language models).
- A drop in error from 3% to 2% might require an order of magnitude more data, compute, or energy.
- Power law scaling implies that many training examples are highly redundant; carefully pruning datasets to retain only informative examples could break power law scaling and achieve exponential scaling instead. ^[inferred]

## Breaking Power Law Scaling

The paper "Beyond neural scaling laws" demonstrates that:
- Under Pareto-optimal pruning strategies, test error can decrease exponentially with the size of the pruned dataset rather than polynomially.
- This requires high-quality pruning metrics that rank examples by their information content.
- The optimal pruning fraction decreases as the initial dataset grows, indicating more aggressive pruning is beneficial for larger datasets.

## Implications

- Simply collecting larger random datasets may be highly inefficient.
- Investment in identifying good pruning metrics could yield larger practical gains than equivalent investment in raw data collection.
- The concept of "foundation datasets"—carefully curated subsets amortized across many training runs—parallels the concept of foundation models.

## Related

- [[concepts/Coreset Selection|Coreset Selection]] — the practice of selecting informative data subsets to improve scaling
- [[references/Data Pruning Paper|Beyond neural scaling laws: beating power law scaling via data pruning]] — the foundational paper on breaking power law scaling
