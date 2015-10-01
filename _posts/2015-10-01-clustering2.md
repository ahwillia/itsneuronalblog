---
layout: post
title: Is clustering mathematically impossible?
comments: True
completed: True
author: alex_williams
topic: Clustering
---

In the [previous post](http://localhost:4000/itsneuronalblog/2015/09/11/clustering1/), we saw intuitive reasons why clustering is a hard,{% include footnote.html n=1%} and maybe even *ill-defined*, problem. In practice, we are often stuck using heuristics that can sometimes perform quite badly when their assumptions are violated (see [*No free lunch theorem*](https://en.wikipedia.org/wiki/No_free_lunch_theorem)). Is there a mathematical way of expressing all of these difficulties? This post will cover some theoretical results of [Kleinberg (2002)](/itsneuronalblog/papers/clustering/Kleinberg_2002.pdf) related to this question.

***Notation.*** Suppose we have a set of $N$ datapoints $x^{(1)}, x^{(2)}, ..., x^{(N)}$. A *clustering function* produces a [*partition*](https://en.wikipedia.org/wiki/Partition_of_a_set) (i.e. a set of clusters), based on the pairwise distances between datapoints. The distance between two points $x^{(i)}$ and $x^{(j)}$ is given by $d(x^{(i)},x^{(j)})$, where $d$ is the *distance function*. We could choose different ways to measure distance,{%include footnote.html n=2 %} for simplicity you can imagine we are using [Euclidean distance](https://en.wikipedia.org/wiki/Euclidean_distance), $\sqrt{ (x^{(i)}-x^{(j)}) \cdot (x^{(i)}-x^{(j)})}$.

### An axiomatic approach to clustering

There are many possible clustering functions we could come up with. Some are stupid &mdash; randomly split the data into two groups &mdash; and others are useful in practice. We would like to precisely define what it means for a clustering function to be "useful in practice."

[Kleinberg (2002)](/itsneuronalblog/papers/clustering/Kleinberg_2002.pdf) proposed that the ideal clustering function would achieve three properties: [*scale-invariance*](/itsneuronalblog/2015/10/01/clustering2/#scale-invariance), [*consistency*](/itsneuronalblog/2015/10/01/clustering2/#consistency), [*richness*](/itsneuronalblog/2015/10/01/clustering2/#richness). The idea is that these principles should align with your intuitive notion of what a "good clustering function" is:

<!--more-->

**1. Scale-invariance:** An ideal clustering function does not change its result when the data are scaled equally in all directions.

{% include image.html url="/itsneuronalblog/img/clustering/clustering-scale-invariance.png" width="500px" title="Scale-invariance." description="For any scalar $\alpha > 0$ the clustering function $f$ produces same result when the distances, $d$, between all datapoints are multiplied: $f(d) = f(\alpha \cdot d)$."%}

**2. Consistency:** If we stretch the data so that the distances between clusters increases and/or the distances within clusters decreases, then the clustering shouldn't change.

{% include image.html url="/itsneuronalblog/img/clustering/clustering-consistency.png" width="500px" title="Consistency." description="Let $d$ and $d^\prime$ be two distance functions. The clustering function produces a partition of points for the first distance function, $d$. If, for every pair $(i,j)$ belonging to the <i>same</i> cluster, $d(i,j) \geq d^\prime(i,j)$, and for every pair belonging to <i>different</i> clusters, $d(i,j) \leq d^\prime(i,j)$ then the clustering result shouldn't change: $f(d) = f(d^\prime)$" %}

**3. Richness:** Suppose a dataset contains $N$ points, but we are not told anything about the distances between points. An ideal clustering function would be flexible enough to produce all possible partition/clusterings of this set. This means that the it automatically determines both the number and proportions of clusters in the dataset. This is shown schemetically below for a set of six datapoints:

{% include image.html url="/itsneuronalblog/img/clustering/clustering-richness.png" width="500px" title="Richness." description="For a clustering function, $f$, richness implies that $\text{Range}(f)$ is equal to all possible partitions of a set of length $N$."%}

### Kleinberg's *Impossibility Theorem*

[Kleinberg's paper](/itsneuronalblog/papers/clustering/Kleinberg_2002.pdf) is a bait-and-switch though. ***It turns out that no clustering function can satisfy all three axioms!*** {%include footnote.html n=3 %} The proof in Kleinberg's paper is a little terse &mdash; A simpler proof is given in [Margareta Ackerman's thesis](http://www.cs.fsu.edu/~ackerman/thesisPhD.pdf), specifically Theorem 21. The intuition provided there is diagrammed below.

{% include image.html url="/itsneuronalblog/img/clustering/impossibility-intuition.png" width="350px" title="Intuition behind impossibility." description="A consequence of the richness axiom is that we can define two different distance functions, $d_1$ (top left) and $d_2$ (bottom right), that respectively put all the data points into individual clusters and into some other clustering. Then we can define a third distance function $d_3$ (top and bottom right) that simply scales $d_2$ so that the minimum distance between points in $d_3$ space is larger than the maximum distance in $d_1$ space. Then, we arrive at a contradiction, since by consistency the clustering should be the same for the $d_1 \rightarrow d_3$ transformation, but also the same for the $d_2 \rightarrow d_3$ transformation."%}

### Clustering functions that satisfy two of the three axioms

The above explanation may still be a bit difficult to digest. Another perspective for understanding the impossibility theory is to examine clustering functions that come close to satisfying the three axioms.

Kleinberg mentions three variants of [single-linkage clustering](https://en.wikipedia.org/wiki/Single-linkage_clustering) as an illustration. Single-linkage clustering starts by assigning each point to its own cluster, and then repeatedly fusing together the nearest clusters (where *nearest* is measured by our specified distance function). To complete the clustering function we need a *stopping condition* &mdash; something that tells us when to terminate and return the current set of clusters as our solution. Kleinberg outlines three different stopping conditions, each of which violates one of his three axioms, while satisfying the other two.

**1. $k$-cluster stopping condition:** Stop fusing clusters once we have $k$ clusters (where $k$ is some number provided beforehand, similar to the [k-means algorithm](https://en.wikipedia.org/wiki/K-means_clustering)).

This clearly violates the *richness* axiom. For example, if we choose $k=3$, then we could never return a result with 2 clusters, 4 clusters, etc. However, it satisfies *scale-invariance* and *consistency*. To check this, notice that the transformations in the above diagrams above do not change which $k$ clusters are nearest to each other. It is only once we start merging and dividing clusters that we get into trouble.

{% include image.html url="/itsneuronalblog/img/clustering/k-stopping-violation.png" width="500px" title="$k$-cluster stopping does not satisfy richness" description=""%}

**2. Distance-$r$ stopping condition:** Stop when the nearest two clusters are farther than a pre-defined distance $r$.

This satisfies *richness* &mdash; we can place $N$ points to end up in $N$ clusters by having the minimum distance between any two points to be greater than $r$, we can place $N$ points to end up in one cluster by having the maximum distance be less than $r$, and we can generate all partitions between these extremes.

It also satisfies *consistency*. Shrinking the distances between points in a cluster keeps the maximum distance less than $r$ (our criterion for defining a cluster in the first place). Expanding the distances between points in different clusters keeps the minimum distance greater than $r$. Thus, the clusters remain the same.

However, *scale-invariance* is violated. If we multiply the data by a large enough number, then the $N$ points will be assigned $N$ different clusters (all points are more than distance $r$ from each other). If we multiply the data by a number close to zero, everything ends up in the same cluster.

{% include image.html url="/itsneuronalblog/img/clustering/distance-r-violation.png" width="500px" title="distance-$r$ stopping does not satisfy scale-invariance" description=""%}

**3. Scale-$\epsilon$ stopping condition:** Stop when the nearest two clusters are farther than a fraction of the maximum distance between two points. This is like the distance-$r$ stopping condition, except we choose $r = \epsilon \cdot \Delta$, where $\Delta$ is the maximum distance between any two data points and $\epsilon$ is a number between 0 and 1.

By adapting $r$ to the scale of the data, this procedure now satisfies *scale-invariance* in addition to *richness*. However, it **does not** satisfy *consistency*. To see this, consider the following transformation of data, in which one cluster (the green one) is pulled much further away from the other two clusters. This increases the maximum distance between data points, leading us to merge the blue and red clusters into one:

{% include image.html url="/itsneuronalblog/img/clustering/clustering-consistency-violation.png" width="500px" title="scale-$\epsilon$ stopping does not satisfy consistency" description=""%}

### Sidestepping impossibility and subsequent work

Kleinberg's analysis outlines what we **should not expect** clustering algorithms to do for us. It is good not to have unrealistic expectations. But can we circumvent his impossibility theorem, and are his axioms even really desirable?

The consistency axiom is particularly suspect as illustrated below:

{% include image.html url="/itsneuronalblog/img/clustering/clustering-consistency-problem.png" width="500px" title="Is consistency a desirable axiom?" description=""%}

The problem is that our intuitive sense of clustering would probably lead us to merge the two clusters in the lower left corner. This criticism is taken up in [Margareta Ackerman's thesis](http://www.cs.fsu.edu/~ackerman/thesisPhD.pdf), which I hope to summarize in a future blog post.

Many clustering algorithms also ignore the *richness* axiom by specifying the number of clusters beforehand. For example, we can run $k$-means multiple times with different choices of $k$, allowing us to re-interpret the same dataset at different levels of granularity. [Zadeh & Ben-David (2009)](http://stanford.edu/~rezab/papers/slunique.pdf) study a relaxation of the richness axiom, which they call $k$-richness &mdash; a desirable clustering function should produce all possible $k$-partitions of a datset (rather than **all** partitions).

Overall, Kleinberg's axiomatic approach provides an interesting perspective on clustering, but his analysis serves more as a starting point, rather than a definitive theoretical characterization of clustering.

{% include sharebar.html %}

#### Footnotes

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} I am using loose language when I say clustering is a "hard problem." Similar to the [previous post](http://localhost:4000/itsneuronalblog/2015/09/11/clustering1/), we will be concerned with why clustering is hard on a conceptual/theoretical level. But it is also worth pointing out that clustering is hard on a computational level &mdash; it takes a long time to compute a provably optimal solution. For example, *k*-means is provably NP-hard for even k=2 clusters [(Aloise et al., 2009)](https://dx.doi.org/10.1007%2Fs10994-009-5103-0). This is because cluster assignment is a discrete variable (a point *either* belongs to a cluster or does not); in many cases, discrete optimization problems are more difficult to solve than continuous problems because we can compute the derivatives of the objective function and thus take advantage of gradient-based methods. (However this [doesn't entirely account for](http://cstheory.stackexchange.com/questions/31054/is-it-a-rule-that-discrete-problems-are-np-hard-and-continuous-problems-are-not) the hardness.)
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Kleinberg (2002) only requires that the distance be nonnegative and symmetric, $d(x_i,x_j) = d(x_j,x_i)$, and not necessarily satisfy the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality). According to Wikipedia these are called [*semimetrics*](https://en.wikipedia.org/wiki/Metric_(mathematics)#Semimetrics). There are many other exotic distance functions that fit within this space. For example, we can choose other [vector norms](https://en.wikipedia.org/wiki/Norm_(mathematics)) $d(x,y) = ||x -y||$ or information theoretic quantities like [*Jensen-Shannon divergence*](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence).
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=3 %} Interesting side note: the title of Kleinberg's paper &mdash; *An Impossibility Theorem for Clustering* &mdash; is an homage to [*Kenneth Arrow's impossibility theorem*](https://en.wikipedia.org/wiki/Arrow%27s_impossibility_theorem), which roughly states that there is no "fair" voting system in which voters rank three or more choices. As in Kleinberg's approach, "fairness" is defined by three axioms, which cannot be simultaneously satisfied.
</p>
