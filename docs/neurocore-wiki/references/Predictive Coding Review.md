---
title: "Predictive Coding: a Theoretical and Experimental Review"
category: references
tags: [neuroscience, predictive-coding, variational-inference, review]
sources: ["docs/papers/Predictive Coding: a Theoretical and Experimental Review.pdf"]
summary: A comprehensive 2022 review of predictive coding theory, covering its mathematical foundations in variational inference, neurobiological plausibility, and relationships to modern machine learning.
provenance:
  extracted: 0.78
  inferred: 0.19
  ambiguous: 0.03
base_confidence: 0.67
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Predictive Coding: a Theoretical and Experimental Review

**Authors**: Beren Millidge, Anil K. Seth, Christopher L. Buckley  
**Institutions**: University of Edinburgh, University of Sussex, CIFAR  
**arXiv**: 2107.12979v4 [cs.AI], July 2022

## Summary

This review provides a comprehensive survey of predictive coding theory, covering its core mathematical structure as variational inference, its neurobiologically plausible microcircuit implementations, and its deep connections to modern machine learning techniques including backpropagation, autoencoders, and active inference.

## Structure of the Review

1. **Predictive Coding as Variational Inference**: Derives predictive coding from first principles of approximate Bayesian inference, showing how minimizing prediction error corresponds to minimizing variational free energy.
2. **Multi-layer Predictive Coding**: Extends the framework to hierarchical architectures with local learning rules.
3. **Dynamical Predictive Coding**: Generalizes to temporal sequences using generalized coordinates of motion (position, velocity, acceleration, etc.).
4. **Precision and Attention**: Discusses how precision (inverse variance) parameters implement attentional modulation and biased competition.
5. **Neurobiological Plausibility**: Reviews empirical evidence for predictive coding in cortical microcircuits.
6. **Relation to Machine Learning**: Explores connections to backpropagation, Kalman filtering, normalizing flows, and VAEs.

## Core Mathematical Framework

The review formalizes predictive coding as minimizing the variational free energy:

$$\mathcal{F} = D_{KL}[q(x|o;\phi) \,||\, p(o,x)]$$

where $q(x|o;\phi)$ is the approximate posterior over latent states and $p(o,x)$ is the generative model. Under Gaussian assumptions, this reduces to a sum of precision-weighted prediction errors at each hierarchical level.

## Key Insights

- **Local Learning**: Synaptic weight updates depend only on local prediction errors and neuronal activities, making predictive coding biologically plausible in a way that backpropagation is not.
- **Precision as Attention**: Precision matrices modulate the influence of prediction errors, providing a computational account of attention that is sensitive to task relevance and global context.
- **Generalized Coordinates**: Representing temporal derivatives explicitly allows predictive coding to handle smoothly changing environments with relatively simple filtering algorithms.
- **Perception as Controlled Hallucination**: Sensory data serves to constrain internally generated predictions rather than directly constructing percepts.

## Connections to Machine Learning

- **Backpropagation**: Predictive coding can approximate backpropagation under certain conditions, suggesting a potential bridge between biologically plausible and standard deep learning.
- **VAEs and Normalizing Flows**: Predictive coding shares structural similarities with variational autoencoders and normalizing flows, though it computes local rather than global losses.
- **Unsupervised Objectives**: The unsupervised predictive loss (reconstructing inputs) is sufficient to develop complex hierarchical representations, paralleling the success of modern autoregressive and contrastive learning methods.

## Relevance to AI

Predictive coding offers principled inspiration for:
- **Data-efficient learning**: Systems that only update on surprising inputs could dramatically reduce training data requirements. ^[inferred]
- **Attention mechanisms**: Precision-weighting provides a normative framework for learned attention that goes beyond heuristic softmax attention. ^[inferred]
- **Continual learning**: Local error computation may help mitigate catastrophic forgetting by preventing global interference. ^[inferred]

## Related

- [[concepts/Predictive Coding|Predictive Coding]] — the concept page
- [[concepts/Coreset Selection|Coreset Selection]] — data pruning inspired by the brain's selective processing
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project drawing on predictive coding for robot data selection
