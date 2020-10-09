---
layout: post
title: A Short Introduction to Optimal Transport and Wasserstein Distance
comments: True
author: alex_williams
completed: True
topic: Optimal Transport
post_description: "An entry point into a topic of great interest in recent ML literature."
---

*This is currently being written... Excuse my hacking.*

<hr>

These notes provide a brief introduction to [optimal transport theory](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)), prioritizing intuition over mathematical rigor. A short and self-contained Python function that computes the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) between two discrete distributions is provided. This demo function shows how to solve a [linear program](https://en.wikipedia.org/wiki/Linear_programming) using the [`scipy.optimize`](https://docs.scipy.org/doc/scipy/reference/optimize.html) library.

### Why Optimal Transport Theory?

A fundamental problem in statistics and machine learning is to come up with useful measures of dissimilarity between pairs of probability distributions.
Concretely, let $P$ and $Q$ be two probability distributions.
Perhaps the most well-known measure of dissimilarity is the [KL divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$D_{KL}(P \| Q) = \int p(x) \log \left ( \frac{p(x)}{q(x)} \right ) dx$$

This notion of dissimilarity is, of course, very useful &mdash; it is a fundamental quantity one encounters in information theory, and in many cases it can be computed or approximated in high-dimensional statistical models.

However, there are also shortcomings to this divergence, which may be important considerations in certain circumstances.
For instance, one of the first things we learn about the KL divergence is that it is not symmetric, $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.
This is arguably not a huge problem, since various symmetrized analogues to the KL divergence exist.
A bigger problem, in many cases, is that the divergence may be infinite if the [support](https://en.wikipedia.org/wiki/Support_(mathematics)) of $P$ and $Q$ are not equal.
Below we sketch three examples of 1D distributions where $D_{KL}(P \| Q) = D_{KL}(Q \| P) = +\infty$.

<img src="/code/ot/1d_schematic.png" width=500>

Intuitively, some of these distribution pairs seem "closer" to each other than others.
But the KL divergence says that they are all infinitely far apart.
One way of circumventing this is to smooth (i.e. add blur to) the distributions before computing the KL divergence, so that the support of $P$ and $Q$ matches.
However, choosing the bandwidth parameter of the smoothing kernel is not always straightforward.

Optimal transport theory helps us construct alternative notions of distance between probability distributions.
In particular, we will encounte the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) (also known as "Earth Mover's Distance" for reasons which will become apparent).
This distance is not only symmetric, but also satisfies the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality) (see, e.g., [Clement & Desch, 2008](https://doi.org/10.1090/S0002-9939-07-09020-X)).
Formally, if $\mathcal{W}(P, Q)$ denotes the Wasserstein distance between two distributions, then we have:

$$
\mathcal{W}(P, Q) = \mathcal{W}(Q, P) \quad\quad \text{and} \quad\quad \mathcal{W}(P, Q) \leq \mathcal{W}(P, M) + \mathcal{W}(M, Q)
$$

for any probability distributions $P$, $Q$, and $M$.
Additionally, it can be shown $\mathcal{W}(P, Q) \neq 0$ unless $P = Q$.
These are all intuitively desirable properties to have in a measure of distance.
Indeed, distance functions that satisfy these properties are called [**metrics**](https://en.wikipedia.org/wiki/Metric_(mathematics)), and they play a foundational role in many areas of mathematics.

Interest in optimal transport seems to have markedly increased in recent years, with applications in imaging ([Lee et al., 2020](https://doi.org/10.1109/TCI.2020.3012954)), generative models ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875)), and biological data analysis ([Schiebinger, 2019](https://doi.org/10.1016/j.cell.2019.01.006)), to name a few.