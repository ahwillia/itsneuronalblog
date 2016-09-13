---
layout: post
title: On the Uniqueness of PCA and Tensor Decompositions
comments: True
author: alex_williams
completed: True
topic: Dimensionality Reduction
post_description: Fuck... My... Life...
---


#### There is no mathematical reason why principal components must be orthogonal
{:.no_toc}

There are two potential justifications for imposing orthogonal components (i.e. $C^T C = I$).

* We tend to plot and think about data in orthogonal coordinate systems. Specifically, imposing orthogonality means the principal components are uncorrelated, so we can talk about the contribution of individual components without referring to the others.
* Imposing this means that there is a unique solution to the optimization problem so that there is a standard protocol for the field.

If the goal was to minimize $\lVert A - W C^T \lVert_F^2$ without constraints, then there are an infinite number of solutions! To see this let $R$ be any invertible $r \times r$ matrix. Then for any candidate solution $\\{W, C\\}$ there is an equally viable solution $\\{W^\prime, C^\prime\\}$ with $W^\prime = W R^{-1}$ and $C^\prime = C R^T$:

$$\lVert A - W C^T \lVert_F^2 = \lVert A - W R^{-1} R C^T \lVert_F^2 = \lVert A - W^\prime C^{\prime T} \lVert_F^2$$

Thus, there is a very general set of linear transformations we can apply to components and loadings without affecting the reconstruction error. This fundamental non-uniqueness is important to understand because it limits our ability to interpret and assign meaning to the components. In other cases, we can understand outcome of the model in this way. For example, suppose we are trying to develop a model of some black-box system that has multiple inputs and multiple outputs. We observe a sequence of inputs $\mathbf{x}$ and their corresponding outputs $\mathbf{y}$:

PICTURE

If the black-box implements a nearly linear transformation, then we can determine this transformation by doing linear regression. The resulting model is a matrix, $B$, which can be used to predict the outcome for any input: $\mathbf{y} = B \mathbf{x}$. Each element in the matrix has a nice interpretation: $B_{ij}$ tells us the effect of input $j$ on output $i$. If $B_{ij}$ is nearly zero then changing input $x_j$ barely effects output $y_i$. If $B_{ij}$ is positive (resp. negative) then increasing $x_j$ increases (resp. decreases) $y_i$. We can even attach units to each $B_{ij}$ as the units of $y_i$ divided by the units of $x_j$.

All of this is very appealing, but falls apart when we move to PCA. In this case, we observe some output of a black-box system, but we don't know the underlying inputs that produced this dataset. For example, we might measure the expression of many genes across many different samples. We may suspect that the input of the system is low-dimensional and more simple that then output we observed (e.g. we might think activity of a handful of transcription factors determines the expression of many genes in our experiment). However, in this hypothetical case, we weren't able to identify or measure these inputs. Nevertheless, there is some real set of inputs $\mathbf{w}^\*_i$ and a real linear transformation $C^\*$ for each observation $\mathbf{a}_i$. Relabeling the input-output diagram we get:

PICTURE

In this case we can consolidate our observations into a data matrix $A$, and do PCA to get a loadings $W$ and components $C$. But it is somewhat intuitive that we can't recover both the inputs and the actual transformation of the system. This is sort of like asking someone to tell you which two numbers you multiplied together to get the number 16 (4 and 4? or 2 and 8?).

However, PCA **does** capture the correct linear subspace.

Perhaps the best known example is [independent components analysis (ICA)](https://en.wikipedia.org/wiki/Independent_component_analysis)
