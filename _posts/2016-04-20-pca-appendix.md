---
layout: post
title: Probabilistic PCA and Factor Analysis
comments: True
author: alex_williams
completed: False
topic: Dimensionality Reduction
post_description: My previous <a href="http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/" target="_blank">post everything you did and didn't know about PCA</a> was poorly named. Clearly, I didn't cover <b><i>everything</i></b> about PCA. I didn't want to, and I still don't.<br><br>But before I moving on to other topics, it is worth nailing down a few points that I glossed over. Jonathan Pillow rightly mentioned that <a href="http://research.microsoft.com/pubs/67218/bishop-ppca-jrss.pdf" target="_blank">probabilistic PCA (pPCA)</a> deserved a deeper treatment, and Matt Kaufman pointed out <a href="https://www.cs.nyu.edu/~roweis/papers/NC110201.pdf" target="_blank">a very nice review</a> by Sam Roweis and Zoubin Ghahramani that I had never come across. I also skipped Factor Analysis, as pointed out by Amy Christensen.
---

### Contents
{:.no_toc}
* TOC
{:toc}

### Introductions

My previous post [*everything you did and didn't know about PCA*](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/) was poorly named. Clearly, I didn't cover ***everything*** about PCA. I didn't want to, and I still don't. But before I moving on to other topics, it is worth nailing down a few points that I glossed over. Jonathan Pillow rightly mentioned that [probabilistic PCA (pPCA)](http://research.microsoft.com/pubs/67218/bishop-ppca-jrss.pdf) deserved a deeper treatment, and Matt Kaufman pointed out [a very nice review](https://www.cs.nyu.edu/~roweis/papers/NC110201.pdf) by Sam Roweis and Zoubin Ghahramani that I had never come across. I also skipped Factor Analysis, as pointed out by Amy Christensen.

I was more or less happy to sweep these topics under the rug, until I came across [a nice post on pPCA](https://liorpachter.wordpress.com/2014/05/26/what-is-principal-component-analysis/) on Lior Pachter's now (in)famous blog, which then lead me to read [Roweis's other paper on pPCA](http://www.cs.nyu.edu/~roweis/papers/empca.pdf).{% include footnote.html n=1 %} All these references brought clarity to my thinking and motivated me to write just a bit more about this. I intended this to be a short post, but it never works out that.

My purpose here is to clarify the implicit probabilistic framework that underlies the optimization framework I took for granted last time. You'll need a passing familiarity with basic multivariate Gaussian distributions to follow along ([*see tutorial here*](https://www.youtube.com/watch?v=eho8xH3E6mE)).

### Univariate least-squares regression $\iff$ Univariate Gaussian noise model

As a warm-up, let's show the equivalence of minimizing squared error and maximizing likelihood under a Gaussian noise model for univariate linear regression. In standard regression notation, we wish to predict $y$ (dependent variable) from $x$ (independent variable) For simplicity, we'll say $x$ and $y$ are scalars and leave it as an exercise to the reader to extend this to multivariate settings. Assume we observe $x$ without noise or error, but our observation $y$ is corrupted by additive Gaussian noise. We observe a sequence of datapoints, $\\{(x_1,y_1), (x_2,y_2), ... , (x_n,y_n)\\}.$ The visual picture to have in mind is this:

PICTURE

Expressing the same picture mathematically:

$$ y_i \overset{\scriptsize\text{i.i.d.}}{\sim} \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2) $$

Said in words: "$y_i$ is a normally distributed random variable with mean $\beta_0 + \beta_1 x_i$ and variance $\sigma^2$." The $\small\text{i.i.d.}$ note above $\sim$ adds a technical but important condition that each observation is ["independent and identically distributed."](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) The probability of each $y_i$ is:

$$ p (y_i \mid x_i, \beta_0, \beta_1) \propto \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right] $$

All we did was plug in $\mu = \beta_0 + \beta_1 x_i$ to the standard Normal distribution (I've ignored the normalizing constant $\frac{1}{\sigma \sqrt{2 \pi}}$, it won't matter to us). Because all observations are $\text{i.i.d.}$, and the joint probability of independent events is the product of their individual probabilities, the aggregate probability of our observations is:

$$
p(\mathbf{y} \mid \mathbf{x}, \beta_0, \beta_1) \propto \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right]
$$

Note that this is function (called the ***likelihood*** of the data) depends on the slope and intercept. The parameters that maximize the likelihood (called the ***maximum likelihood estimate***) are found by solving:

$$
\begin{aligned}
& \underset{\beta_0, \beta_1}{\text{maximize}}
& & \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right ]
\end{aligned}
$$

Our goal was to show that the solution to this optimization problem is equivalent to finding the slope and intercept that minimize the sum of squared residuals. This is simple to do, since maximizing likelihood is equivalent to minimizing the negative log-liklihood{% include footnote.html n=2 %}. The rest requires basic algebraic manipulations.

$$
\begin{aligned}
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & -\log \left [ \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right ] \right ] \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  - \log \left [ \exp \left[ - \frac{(y_i-\hat{y}_i)^2}{2 \sigma^2 } \right ] \right ] \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \frac{1}{2 \sigma^2 } \sum_{i=1}^n  (y_i-\hat{y}_i)^2 \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  (y_i - \hat{y}_i)^2 \\
\end{aligned}
$$

Where the final line follows since $\sigma^2$ is a constant, and rescaling the objective function does not effect the optimal values of $\beta_1$ and $\beta_0$.

### Probabilistic PCA $\iff$ Istropic Gaussian noise model 

The motivation of PCA is not to predict a set of dependent variable from independent variables. Instead, all observed variables are treated on an equal footing, and noise is present in all variables (not just the dependent variables). Following similar notation from the last post, let $\mathbf{a}_{i} = \\{ a_1, a_2, ..., a_p \\}$ denote the observation $i$; each observation consists of $p$ measured features. As always, we assume the data is mean-centered &mdash; i.e. each feature has zero mean across the dataset.

PCA assumes that the data are described by a collection of $r$ latent factors. Each observation is formed by a linear combination of the $r$ factors ("principal components") with weights $\mathbf{w}_i$ ("loadings"), plus a noise term $\boldsymbol{\xi} \in \mathbb{R}^p$.

$$ \mathbf{a} = w_1 \mathbf{c}_1 + w_2 \mathbf{c}_2 + ... + w_r \mathbf{c}_r + \boldsymbol{\xi}$$

We have again used similar notation so that the $k$th principal component is denoted by $\mathbf{c}_{k} = \\{ c_1, ..., c_p \\}$ and the loadings/weights are given by $\mathbf{w} = \\{ w_1, ..., w_r \\}$. This above equation describes a single observation. I abused notation by dropping the index $i$ to denote the $i$th observation. However, we will assume every observation is drawn $\small\text{i.i.d.}$ so this index is not too important. If you like, we can also represent the full dataset as matrix equation.{%include footnote.html n=3 %}

Our derivation of probabilistic PCA will assume the loadings are drawn from a standard, multivariate normal distribution and that $\boldsymbol{\xi}$ follows a zero-mean, isotropic Gaussian distribution:

$$ \mathbf{w} \overset{\scriptsize\text{i.i.d.}}{\sim} \mathcal{N}(0,I_r) \quad \quad \boldsymbol{\xi} \overset{\scriptsize\text{i.i.d.}}{\sim} \mathcal{N}(0,\sigma^2 I_p) $$

where $I_r$ denotes an $r \times r$ identity matrix, and $I_r$ denotes a $p \times p$ identity matrix. These assumptions have an important consequence and interpretation. Think of $\mathbf{w}$ and $\boldsymbol{\xi}$ as random Gaussian inputs to a fixed linear system (which is given by the components $\mathbf{c}_k$). Since these inputs are Gaussian, and the system is linear, then the output of the system (i.e. our data, $\mathbf{a}$) is also Gaussian distributed:

$$
\mathbf{a} \sim N(0, C C^T + \sigma^2 I_p )
$$

[Click here to see a derivation of this result](http://math.stackexchange.com/questions/332441/affine-transformation-applied-to-a-multivariate-gaussian-random-variable-what). As in the last post, the columns of $C \in \mathbb{R}^{p \times r}$ contain the principal components, $C = \[ \mathbf{c}_1 ... \mathbf{c}_r \]$. The vectors $\mathbf{a}$ and $\mathbf{w}$ are row vectors, see [\[3\]](#f3b).

Intuitively, this equation tells us that the inputs (spherical gaussian noise) are projected onto an $r$-dimensional subspace, and stretched/rotated by a linear transform,

$$
\mathbf{a} \sim N(0, C C^T + \sigma^2 I_p )
$$

Before digging deeper into the math, let's visualize this basic picture in two dimensions. This should be compared the graphical depiction of regression in figure 1.

PICTURE

LEGEND: Probabilistic PCA $p=2$ and $r=1$.

http://math.stackexchange.com/questions/332441/affine-transformation-applied-to-a-multivariate-gaussian-random-variable-what

To express this mathematically, we use the multivariate Normal probability distribution:

$$
\left[
\begin{array}{c}
    x_i \\ y_i
\end{array}
\right]
\sim
N \left ( w_i \left[ \begin{array}{c} c_x \\ c_y \end{array} \right],
\left [ \begin{array}{cc} \sigma^2 & 0 \\ 0 &\sigma^2 \end{array} \right ] \right )
$$

As in the linear regression warmup, we observe a bunch of datapoints $(x_i, y_i)$ and we want to find parameters that maximize the likelihood. We need to estimate two quantities, the direction of the line $[c_x c_y]$ and, for each datapoint, a scalar $w_i$ which tells us how far along that line that datapoint truly sits. The covariance matrix is diagonal and controlled by a single parameter $\sigma^2$ (the variance in $x$ and $y$); this corresponds to our assumption that noise is uncorrelated and equal in magnitude for $x$ and $y$.

ANOTHER PICTURE

Now that we've visualized this for two dimensions, let's go to 

$$
\left[
\begin{array}{c}
    x_i \\ y_i
\end{array}
\right]
\sim
N \left (  \mathbf{w}_i C^T ,
\left [ \begin{array}{cc} \sigma^2 & 0 \\ 0 &\sigma^2 \end{array} \right ] \right )
$$

parameter $\sigma^2$ scales the noise, and 

$$
math
$$

define $[_x c_y]$

Mathematically, we can express this as follows

$$
\begin{aligned}
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & -\log \left [ \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right ] \right ] \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  - \log \left [ \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right ] \right ] \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \frac{1}{2 \sigma^2 } \sum_{i=1}^n  (y_i-(\beta_0 + \beta_1 x_i))^2 \\
& \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  (y_i - \hat{y}_i)^2 \\
\end{aligned}
$$

### What is the difference between probabilistic PCA and PCA?

Great question. Actually, if I may inject a short opinionated rant, I think this is explained terribly in pretty much every published paper I've read.{% include footnote.html n=4 %} The best explanation in my view is [this stackexchange answer](http://stats.stackexchange.com/questions/123063/is-there-any-good-reason-to-use-pca-instead-of-efa/123136#123136). Also see the explanation in [Kevin Murphy's textbook on Machine Learning](https://www.cs.ubc.ca/~murphyk/MLbook/).

The short answer is that they are basically the same. They don't deserve different names. It just adds confusion.

The longer answer is that both pPCA and PCA recover the same linear subspace of $W$ and $C$, but pPCA shrinks the estimate of the loadings (i.e. $\mathbf{w}_i$ for each observation) towards zero.

### Logistic PCA, Robust PCA, etc.

In the previous post I enumerated several el

### Noise in pPCA and Factor Analysis

Factor analysis makes a weaker assumption that the noise is Gaussian and uncorrelated.

> "Principal Component Analysis" is a dimensionally invalid method that gives people a delusion that they are doing something useful with their data. If you change the units that one of the variables is measured in, it will change all the "principal components"! It's for that reason that I made no mention of PCA in my book. I am not a slavish conformist, regurgitating whatever other people think should be taught. I think before I teach. David J C MacKay.

http://blog.explainmydata.com/2012/07/should-you-apply-pca-to-your-data.html

### Factor Analysis vs. PCA on synthetic data



### An opinionated conclusion: stop over-emphasizing probability

I'm sure if you live and breath Bayesian statistics the notation and language.

, but the notation is bloated and pretty much terrible in every other way you can think of. Even if notation was equal (its really not) the optimization viewpoint is conceptually much simpler in my view. Undergraduates in biology and social sciences are taught to "minimizing (squared) residuals" and not "maximizing likelihood under a Gaussian"


#### Other references:

* [StackExchange: PCA vs. Factor Analysis on the same dataset](http://stats.stackexchange.com/questions/94048/pca-and-exploratory-factor-analysis-on-the-same-dataset-differences-and-similar)


{% include sharebar.html %}

#### Footnotes
{:.no_toc}

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} Roweis refers to probabilistic PCA as *sensible PCA* in his paper. He published concurrently with Tipping and Bishop, and their name has mostly won out.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Applying $\log$ doesn't change the optimization problem because it is a monotonic function. Maximizing a function is equivalent to minimizing that function multiplied by $-1$.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=3 %} The matrix equation matches my last blog post, $ A = W C^T + R $. In this post $\mathbf{a}$ is a row in $A$, $\mathbf{w}$ is a row in $W$, and $\mathbf{c}$ is a column in $C$ (i.e. a row in $C^T$). The matrix $R \in \mathbb{R}^{n \times p}$ is the noise term: each $\boldsymbol{\xi}$ is a row in $R$.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=4 %} It's possible that there are more good explanation out there. Maybe I've just been unlucky with the papers I've read.
</p>
