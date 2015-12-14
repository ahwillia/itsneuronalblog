---
layout: post
title: What is a neuron's firing rate? Is it even well-defined?
comments: True
completed: False
author: alex_williams
topic: Neural Coding
---

Finally, a post on real neuroscience!

This post is a quick commentary on a recent paper from Asohan Amarasingham, Stuart Geman, and Matthew T. Harrison: [*"Ambiguity and nonidentifiability in the statistical analysis of neural codes."*](http://dx.doi.org/10.1073/pnas.1506400112) The basic result is evident

{% include image.html url="/itsneuronalblog/img/coding/doubly-stochastic-nonidentifiability.jpg" width="800px" %}

In other words:

{% include image.html url="/itsneuronalblog/img/memes/one-does-not-simply-firing-rate.jpg" width="500px" %}

[The paper](http://dx.doi.org/10.1073/pnas.1506400112) is excellently written and very approachable, so I'll let it speak for itself for the most part. I'll focus the rest of this post on more interesting questions: How does this impact previously published results? How should it inform future work? Does it require that we radically re-think basic principles of neural coding?

<!--more-->

### Outline

* Churchland vs. Pillow
* Ashok and Friedamann's work
	* Larry & Mark's overview
* Boerlin/Deneve work

#### Acknowledgements

Chris 

#### Footnotes

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} It is instructive to prove that the arithmetic mean (i.e. centroid) of a set of points [minimizes the sum-of-squared residuals](http://math.stackexchange.com/questions/967138/formal-proof-that-mean-minimize-squared-error-function). Similarly, the median [minimizes the sum-of-absolute residuals](http://math.stackexchange.com/questions/113270/the-median-minimizes-the-sum-of-absolute-deviations).
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Finding the solution to the k-means optimization problem is known to be NP-hard [(Aloise et al., 2009)](https://dx.doi.org/10.1007%2Fs10994-009-5103-0). Note that there are simple and efficient algorithms that find local minima, given an initial guess. However, solving the problem (i.e. finding and certifying that you've found the global minimum) is NP-hard in the worst-case.
</p>
