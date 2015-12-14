---
layout: post
title: Is backprop necessary to train neural networks? 
comments: True
completed: False
author: alex_williams
topic: Artificial Networks
---

Introductory remarks here....

<!--more-->

### Linear network

Consider a two-layer linear neural network (perhaps make an illustration?):

$$ \mathbf{h} = A \mathbf{x} $$

$$ \mathbf{y} = W \mathbf{h} $$

We have a target output $\tilde{\mathbf{y}} = T \mathbf{x}$

Using a quadratic loss function, the error is $\mathbf{e} = \frac{1}{2} \big\vert\big\vert \tilde{\mathbf{y}} - \mathbf{y} \big\vert\big\vert^2_2$

We would like to do gradient descent, which means we update $A$ and $W$ according to:

$$ \Delta W = -\eta \frac{\partial \mathbf{e}}{\partial W} $$

$$ \Delta A = -\eta \frac{\partial \mathbf{e}}{\partial A} $$

For $W$ we compute the gradient directly

$$ \frac{\partial}{\partial W} \frac{1}{2} \big\vert\big\vert \tilde{\mathbf{y}} - \mathbf{y} \big\vert\big\vert^2_2 = (\tilde{\mathbf{y}} - \mathbf{y}) \frac{\partial}{\partial W} \mathbf{y} = (\tilde{\mathbf{y}} - \mathbf{y}) \mathbf{h} $$

#### Acknowledgements

Freidamann, Ben

#### Footnotes

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} It is instructive to prove that the arithmetic mean (i.e. centroid) of a set of points [minimizes the sum-of-squared residuals](http://math.stackexchange.com/questions/967138/formal-proof-that-mean-minimize-squared-error-function). Similarly, the median [minimizes the sum-of-absolute residuals](http://math.stackexchange.com/questions/113270/the-median-minimizes-the-sum-of-absolute-deviations).
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Finding the solution to the k-means optimization problem is known to be NP-hard [(Aloise et al., 2009)](https://dx.doi.org/10.1007%2Fs10994-009-5103-0). Note that there are simple and efficient algorithms that find local minima, given an initial guess. However, solving the problem (i.e. finding and certifying that you've found the global minimum) is NP-hard in the worst-case.
</p>
