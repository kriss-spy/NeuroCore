---
title: Predictive Coding
category: concepts
tags: [neuroscience, brain-inspired-ai, variational-inference, attention]
sources: ["docs/papers/Predictive Coding: a Theoretical and Experimental Review.pdf", README.md]
summary: A unifying theory of cortical function proposing that the brain minimizes prediction errors through hierarchical generative models, with strong ties to variational inference and attention.
provenance:
  extracted: 0.72
  inferred: 0.24
  ambiguous: 0.04
base_confidence: 0.77
lifecycle: draft
lifecycle_changed: 2026-05-08
created: 2026-05-08T13:30:00+08:00
updated: 2026-05-08T13:30:00+08:00
---

# Predictive Coding

Predictive coding is an influential theory in computational and cognitive neuroscience that postulates the core function of the brain is to minimize prediction error with respect to a generative model of the world. Rather than passively processing all incoming sensory data, the brain constantly generates top-down predictions and only allocates significant resources to stimuli that deviate from those predictions.

## Key Ideas

- The brain is organized as a hierarchy of layers, where each layer predicts the activity of the layer immediately below it.
- **Prediction errors**—the mismatch between predicted and actual input—are transmitted upward through the hierarchy and used to update both immediate inferences (fast timescale) and synaptic weights (slow timescale).
- This framework can be formalized as **variational inference** under Gaussian generative models, where minimizing prediction error corresponds to minimizing variational free energy.
- Predictive coding naturally explains perceptual phenomena such as end-stopping, bistable perception, repetition suppression, illusory motion, and attentional modulation.

## Mathematical Framework

Under the variational formulation:
- The brain maintains an approximate posterior $q(x|o; \phi)$ over latent states $x$ given observations $o$.
- Inference proceeds by minimizing the variational free energy $\mathcal{F} = D_{KL}[q(x|o;\phi) || p(o,x)]$, which upper-bounds the negative log-evidence.
- The free energy decomposes into an **accuracy** term (likelihood of observations under the generative model) and a **complexity** term (divergence from the prior).

## Precision and Attention

A critical extension of predictive coding is the notion of **precision** (inverse variance):
- Precision parameters multiplicatively modulate the influence of prediction errors, effectively controlling the "signal-to-noise ratio" of sensory updates.
- Precision has been linked to **attentional modulation**: top-down contextual factors (e.g., task relevance) can increase the precision of specific prediction errors, amplifying their impact on inference. ^[inferred]
- This provides a principled mechanism for the brain's ability to filter background noise and focus on behaviorally relevant stimuli—functionally analogous to the **Reticular Activating System (RAS)**.

## Temporal Predictive Coding

For dynamic sequences, predictive coding can be extended using **generalized coordinates of motion**, where the brain represents not only the current state but also its velocity, acceleration, and higher-order derivatives. This allows the system to track smoothly changing environments while retaining the static analytical machinery of equilibrium solutions.

## Relation to Machine Learning

- Predictive coding shares deep connections with **autoencoders**, **variational autoencoders (VAEs)**, and **normalizing flows**.
- The learning dynamics in predictive coding networks can be implemented using local, biologically plausible **Hebbian learning rules**, contrasting with the non-local backpropagation algorithm.
- Nevertheless, recent work has shown that predictive coding can approximate backpropagation of error under certain conditions, bridging the gap between biologically plausible learning and deep learning.

## Brain-Inspired Applications

Predictive coding provides a natural foundation for designing data-efficient AI systems:
- **Temporal Filtering**: Just as the brain ignores predictable sensory frames, machine learning systems can filter temporally redundant training examples based on prediction error magnitude. ^[inferred]
- **Attention-Based Pruning**: Precision-weighting suggests that task-relevant features should receive higher effective sample weights, informing coreset selection strategies. ^[inferred]

## Related Concepts

- [[concepts/Coreset Selection|Coreset Selection]] — data pruning inspired by the brain's selective attention mechanisms
- [[projects/NeuroCore/NeuroCore|NeuroCore]] — a project applying predictive coding principles to VLA training data selection
- [[references/Predictive Coding Review|Predictive Coding: a Theoretical and Experimental Review]] — the comprehensive review paper
