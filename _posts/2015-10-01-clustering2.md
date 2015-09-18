---
layout: post
title: What is the state of theory on clustering?
comments: True
author: alex_williams
---

In the [previous post](http://localhost:4000/itsneuronalblog/2015/09/11/clustering1/), we saw intuitive reasons why clustering is a hard<sup>[1]</sup>, and maybe even *ill-defined*, problem. In practice, we are often stuck using heuristics that can sometimes perform quite badly when their assumptions are violated (see [*No free lunch theorem*](https://en.wikipedia.org/wiki/No_free_lunch_theorem)). Over the years people have come up with [many](/), [many](/) [different](/) [heuristics](/). Are there any common, unifying theoretical principles that can help us navigate this problem? This post will summarize a nice series of papers on this topic.

### A formal statement of the clustering problem

We have a set of points \(x_1,x_2,x_3, ... ,x_n\)

### Kleinberg's *Impossibility Theorem*

[Kleinberg (2002)](/itsneuronalblog/papers/Kleinberg_2002.pdf) proposed that we consider an *axiomatic approach* to clustering. That is, we very precisely define a set of principles (*axioms*) that the ideal clustering algorithm would achieve. Kleinberg proposed three such properties: *scale-invariance

### Relaxing the consistency axiom

Ackerman & Ben-David (2008)

My hotel 

{% include image.html url="/itsneuronalblog/img/clustering/clustering-consistency-problem.png" width="500px" title="Is consistency a desirable axiom?" description=""%}

### Relaxing the richness axiom

(Zadeh & Ben-David, 2009)


Papers:

* http://www.quick2degrees.com/ddata/315.pdf
* http://www.researchgate.net/profile/Shai_Ben-David/publication/228941037_Stability_of_k-means_clustering/links/0c9605249d9f25ad8d000000.pdf
* http://biostatistics.oxfordjournals.org/content/early/2006/04/12/biostatistics.kxj029.full.pdf
* http://amstat.tandfonline.com/doi/abs/10.1198/016214508000000454
* http://www.mitpressjournals.org/doi/pdf/10.1162/089976604773717621
* https://uwspace.uwaterloo.ca/bitstream/handle/10012/6824/Ackerman_Margareta.pdf?sequence=1

Figures:

{% include image.html url="https://upload.wikimedia.org/wikipedia/commons/3/32/Set_partitions_4%3B_Hasse%3B_circles.svg" width="400px" title="All partition refinements for four datapoints." description="At the top of the diagram, we begin with a partition containing a single cluster that all datapoints belong to (red). Sequential refinements lead to the bottom partition, in which each point belongs to its own cluster. Each <b><i>vertical</i></b> path among the layers (no turning!) produces an <i>anti-chain</i>. By Watchduck (a.k.a. Tilman Piesk) <a href='http://www.gnu.org/copyleft/fdl.html'>GFDL</a> or <a href='http://creativecommons.org/licenses/by/3.0'>CC BY 3.0</a>, via <a href='https://commons.wikimedia.org/wiki/File%3ASet_partitions_4%3B_Hasse%3B_circles.svg'>Wikimedia Commons</a>"%}


<span class="footnotes">
**[1]** I am using loose language when I say clustering is a "hard problem." Similar to the [previous post](http://localhost:4000/itsneuronalblog/2015/09/11/clustering1/), we will be concerned with why clustering is hard on a conceptual/theoretical level. But it is also worth pointing out that clustering is hard on a computational level &mdash; it takes a long time to compute a provably optimal solution. For example, *k*-means is provably NP-hard for even k=2 clusters [(Aloise et al., 2009)](https://dx.doi.org/10.1007%2Fs10994-009-5103-0). This is because cluster assignment is a discrete variable (a point *either* belongs to a cluster or does not); in many cases, discrete optimization problems are more difficult to solve than continuous problems because we can compute the derivatives of the objective function and thus take advantage of gradient-based methods. (However this [doesn't entirely account for](http://cstheory.stackexchange.com/questions/31054/is-it-a-rule-that-discrete-problems-are-np-hard-and-continuous-problems-are-not) the hardness.)
</span>
