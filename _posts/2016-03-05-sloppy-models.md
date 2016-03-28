---
layout: post
title: Sloppy Models #cosyne16
comments: True
author: alex_williams
completed: False
topic: Dimensionality Reduction
---

# Ben Machta

* Water is not a simple liquid but solute motion is described by the very simple diffusion equation
* Signal cascade model (PC12, NGF) complaints
	* Models are way too simple -- real biology is high-dimensional (biologists)
	* Models are way too complex -- prediction space is low-dimensional (physicists)
* What is sloppiness?
	* Anisotropic parameter uncertainty / 'hyper-ribbon' structure
	* Anisotropy in Hessian
* Simpler model for theory -- radioactive decay (least-squares fit)
* Embed possible model parameters into the space of predictions
	* hyper-ribbon structure animation
	* Why? Because constraining y(t) at some point highly constrains possible behaviors
* Microscopic models are sloppy after coarsening

* Despite constant plasticity, firing rates preserved over days

{% include sharebar.html %}

#### References

* P. Baldi and K. Hornik. Neural networks and principal component analysis: Learning from examples without local minima. Neural Networks, 2(1):53–58, 1989.

#### Footnotes


<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} There are many interesting remarks to make for the aficionados. Note that the covariance matrix $Q = A^T A$ is a symmetric, [positive semi-definite matrix](https://en.wikipedia.org/wiki/Positive-definite_matrix). This means that $\mathbf{z}^T Q \mathbf{z} \geq 0$ for any vector $\mathbf{z}$, and equivalently that all eigenvalues of $Q$ are nonnegative. PCA maximizes $\mathbf{w}^T Q \mathbf{w}$; the [solution to this problem](http://math.stackexchange.com/questions/23596/why-is-the-eigenvector-of-a-covariance-matrix-equal-to-a-principal-component) is to set $\mathbf{w}$ to the eigenvector of $Q$ associated with the largest eigenvalue. All [symmetric matrices have orthogonal eigenvectors](http://math.stackexchange.com/questions/82467/eigenvectors-of-real-symmetric-matrices-are-orthogonal), which is why the principal component vectors are always orthogonal. PCA could be achieved by doing an [eigendecomposition](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix) of the covariance matrix. <br><br>Even better, instead of computing $A^T A$ and then an eigendecomposition, one can directly compute the [singular-value decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition) directly on the raw data matrix. SVD works for non-square matrices (unlike eigendecomposition) and produces $A = U S V^T$ where $S$ is a diagonal matrix of [singular values](https://en.wikipedia.org/wiki/Singular_value) and $U$ and $V$ are [orthogonal matrices](https://en.wikipedia.org/wiki/Orthogonal_matrix). Since the transpose of an orthogonal matrix is its inverse, and basic properties of the transpose operator: $A^T A = V S U^T U S V^T = V S S V^T =  V \Lambda V^T$, where $\Lambda$ is just a diagonal matrix of eigenvalues, which are simply the squared singular values in $S$. Thus, doing the SVD on the raw data directly gives you the eigendecomposition of the covariance matrix.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Drop it into conversation at parties. You'll sound smart and not at all obnoxious.
</p>
<p>
Check out Appendix A of [Madeleine Udell's thesis](https://courses2.cit.cornell.edu/mru8/doc/udell15_thesis.pdf), which showcases five equivalent formulations of PCA as optimization problems.
</p>
