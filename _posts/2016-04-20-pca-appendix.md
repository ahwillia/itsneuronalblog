---
layout: post
title: Demystifying Factor Analysis
comments: True
author: alex_williams
completed: False
topic: Dimensionality Reduction
post_description: Long time no blog. My last post <a href="http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/" target="_blank">post everything you did and didn't know about PCA</a> was poorly named. Long time no blog. My last post <a href="http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/"><i>everything you did and didn't know about PCA</i></a> was poorly named. Clearly, I didn't cover <b>everything</b> about PCA. In particular, I think I may have offended statisticians by not using probability, and not mentioning noise models. <br><br> I recently heard from an esteemed person that adding an explicit noise model to PCA helps tremendously in practice. I was skeptical. So let's explore that claim.

---

### Contents
{:.no_toc}
* TOC
{:toc}

### Introduction

Long time no blog. My last post [*everything you did and didn't know about PCA*](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/) was poorly named. Clearly, I didn't cover ***everything*** about PCA. In particular, I think I may have offended statisticians by not using probability, and not mentioning noise models.

I recently heard from an esteemed person that adding an explicit noise model to PCA helps tremendously in practice. I was skeptical. So let's explore that claim.

Here, I'll show that [probabilistic PCA](#) is equivalent to PCA with quadratic regularization and that [factor analysis](#) is equivalent to ??.

### Noise Models

So what is a *noise model* anyways? It is easiest to explain this for the case of linear regression, where we predict a dependent variable, $y$, from an independent variable, $x$ (both scalars, for now). Then the linear regression model is:

$$
\begin{equation}
y_i = \beta_1 x_i + \beta_0 + \epsilon \, , \quad \epsilon \sim \text{Normal}(0, \sigma^2)
\end{equation}
$$

Here, $\beta_1$ and $\beta_0$ are the slope and intercept of the regression line &mdash; the parameters of our model that we want to estimate from data. Each datapoint or observation is indexed by $i = 1, 2, ..., n$. The [random variable](https://en.wikipedia.org/wiki/Random_variable) $\epsilon$ adds noise to every observation. The classic noise model here is a Normal (Gaussian) distribution, with variance $\sigma^2$. The visual picture to have in mind is this:

PICTURE

But we can generalize linear regression by changing the noise model. For example, we could assume that $\epsilon$ follows a [Laplacian distribution](https://en.wikipedia.org/wiki/Laplace_distribution), which contains [heavier tails](https://stats.stackexchange.com/questions/180952/t-distribution-having-heavier-tail-than-normal-distribution) than the Normal distribution. This makes the noise model more accomodating to outliers, and ends up leading to a flavor of [robust regression](https://en.wikipedia.org/wiki/Robust_regression).
So changing the noise model can help us develop other useful models for different applications.

One more quick example. Both Gaussian and Laplacian noise models add 
By changing to a Poisson noise model we can accomodate [count data](https://en.wikipedia.org/wiki/Count_data) more naturally. The Poisson distribution places probability only on nonnegative, integer values of $y_i$. For example, the number of neuron spikes within a small time window can be modeled as a Poisson random variable. The visual picture to have in mind is this:

PICTURE

This is called [Poisson regression](https://en.wikipedia.org/wiki/Poisson_regression). All of these models can be beautifully unified into a class of models called [*generalized linear models* (GLMs)](https://en.wikipedia.org/wiki/Generalized_linear_model), which have become quite popular in neuroscience (CITATIONS).

Thus, different noise distributions intuitively lead to different models. But how, precisely, does this happen? We'll see that the noise model directly changes the optimization problem that you solve to fit the model parameters. In the language of my last post, it changes the *loss* of the objective function.

### Least-squares regression $\iff$ Gaussian noise model

As a warm-up, let's prove the following:

> In linear regression, fitting a model to minimize the sum of squared residuals, is equivalent to fitting a model to maximize the likelihood function under a Gaussian noise model.

Equation 1 means that each $y_i$ is a normally distributed random variable with mean $\beta_0 + \beta_1 x_i$ and variance $\sigma^2.$ We treat $\sigma^2$ as a constant. Formally,

$$ y_i \overset{\scriptsize\text{i.i.d.}}{\sim} \mathcal{N}(\beta_0 + \beta_1 x_i, \sigma^2) $$

The symbol $\overset{\scriptsize\text{i.i.d.}}{\sim}$ means each observation is ["independent and identically distributed."](https://en.wikipedia.org/wiki/Independent_and_identically_distributed_random_variables) From the definition of the Normal distribution, we have:

$$ p (y_i \mid x_i, \beta_0, \beta_1) \propto \exp \left[ - \frac{(y_i-\hat{y}_i)^2}{2 \sigma^2 } \right] $$

where $\hat y_i = \beta_0 + \beta_1 x_i$ is our linear model estimate. Each $y_i$ is an independent event (due to the $\text{i.i.d.}$ assumption) so we simply multiply these terms together to get the joint probability over *all* $y_i$:

$$
L(\beta_0, \beta_1) = p(\mathbf{y} \mid \mathbf{x}, \beta_0, \beta_1) \propto \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right]
$$

This function, $L(\beta_0, \beta_1)$, is called the ***likelihood*** of the data.{%include footnote.html n=1 %} Our goal is to find the parameters (slope $\beta_0$ and intercept $\beta_1$) that produce the [***maximum likelihood***](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation).

$$
\begin{aligned}
& \underset{\beta_0, \beta_1}{\text{maximize}}
& & L(\beta_0, \beta_1)
\end{aligned}
$$

In practice, we find the maximum likelihood estimates by *minimizing* the *negative log-likelihood* (these are equivalent problems). Taking the logarithm affords us [numerical stability](https://stats.stackexchange.com/questions/174481/why-to-optimize-max-log-probability-instead-of-probability), and most general-purpose optimization packages are built to minimize functions rather than maximize them. Now the rest of the proof follows from basic algebraic manipulations.

$$
\begin{aligned}
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & -\log L(\beta_0, \beta_1) \\
& \quad \quad \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & -\log \left [ \prod_{i=1}^n  \exp \left[ - \frac{(y_i-(\beta_0 + \beta_1 x_i))^2}{2 \sigma^2 } \right ] \right ] \\
& \quad \quad \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  - \log \left [ \exp \left[ - \frac{(y_i-\hat{y}_i)^2}{2 \sigma^2 } \right ] \right ] \\
& \quad \quad \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \frac{1}{2 \sigma^2 } \sum_{i=1}^n  (y_i-\hat{y}_i)^2 \\
& \quad \quad \Big \Updownarrow \\
& \underset{\beta_0, \beta_1}{\text{minimize}}
& & \sum_{i=1}^n  ( y_i - \hat{y}_i )^2\\
\end{aligned}
$$

Where the final line follows since we can rescale the objective function by $1 / 2 \sigma^2$ without changing the values of $\beta_1$ and $\beta_0$ that are best fit to the data. 

**Notice!** The amount of noise ***<u>does not affect</u>*** the best fit slope and intercept. That is, changing the value of $\sigma^2$ does not change the maximum likelihood estimates of $\beta_0$ and $\beta_1$.

### A noise model for PCA

Linear regression seems to be taught to undergraduates as finding the line that minimizes the squared residuals. This makes sense to me &mdash; it is easy to teach and understand. We have now seen that this is the same as maximizing the likelihood under a *univariate* (LINK) Gaussian noise model. It is really nice to know this connection because it helps to understand [generalized linear models (GLMs)](https://en.wikipedia.org/wiki/Generalized_linear_model), which we've seen are a very flexible and useful extension to classic regression. However, I wouldn't say the former interpretation of linear regression is wrong or inferior.

I will now advance a similar argument for PCA. Classically, PCA is introduced as a technique that *maximizes variance* or (equivalently) *minimizes squared residuals* as I outlined in [my last post](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/). However, you can also think of PCA as a maximum likelihood model under an *isotropic*, *multivariate* Gaussian noise model (LINKS).

In essence, PCA generalizes linear regression by assuming there is noise in both $x$ and $y$ (rather than just in $y$). Furthermore, the noise is *isotropic*, meaning that $x$ and $y$ are equally noisy. For the two-dimensional case, the picture to have in mind is this:

PICTURE

Now let's state this picture formally. Because $x$ and $y$ are equally noisy, there is no longer a solid distinction between the independent and dependent variables.  Thus, we will change notation so that all measured variables/datapoints are collected into a matrix $\mathbf{X}$. Each row of $\mathbf{X}$ is an observed datapoint, so for the 2-dimensional case sketched above $\mathbf{X}$ has two columns. Of course, we are interested in the higher-dimensional case, when $\mathbf{X}$ has many columns.

The PCA model is:

$$
\mathbf{X} = \mathbf{U} \mathbf{V}^T + \mathbf{E}
$$

where $\mathbf{E}$ is a Gaussian random matrix, with each element randomly drawn according to:

$$
E_{ij} \overset{\scriptsize\text{i.i.d.}}{\sim} \text{Normal}(0 \, , \, \sigma^2)
$$

We can also re-express this equation for a single row of $\mathbf{X}$. The PCA model for row $i$ of $\mathbf{X}$ is:

$$
\mathbf{x}_{i:} \overset{\scriptsize\text{i.i.d.}}{\sim} \text{MvNormal} \left ( \hat{\mathbf{x}}_{i:} \, , \, \sigma^2 \mathbf{I} \right )
$$

Where the estimate is:

$$
\hat{x}_{i:} = \sum_{r=1}^R u_{ir} \mathbf{v}_{:r}
$$

As before, our goal is to find model parameters that maximize the likelihood function. Here, the likelihood function is a product of multivariate Gaussians:

EQUATION



In fact, this is how experts viewed PCA until two papers by [Roweis (1998)](http://papers.nips.cc/paper/1398-em-algorithms-for-pca-and-spca.pdf) and [Tipping \& Bishop (1999)](http://dx.doi.org/10.1111/1467-9868.00196) drew the connection between PCA and a Gaussian noise model.

Both authors insist on giving new names to their new model. Roweis calls it "Sensible PCA" and Tipping/Bishop call it "Probabilistic PCA". I think the new names aren't necessary. I particularly don't like "Sensible PCA" as it suggests that PCA in the classic sense is *not sensible*.

Opinions aside, what does PCA look like with noise? It turns out to actually looks a lot like the regression example except with Gaussian noise incorporated into $x$ as well as $y$. Because both variables contain noise, the distinction between dependent and independent variables disappears, so we change notation. Each datapoint is a vector in a high dimensional space, which we'll denote $\mathbf{x}_i$. Our full dataset is collected into a matrix $\mathbf{X}$. Each $\mathbf{x}_i$ is a row of $\mathbf{X}$.

PCA finds a low-dimensional space characterized by loadings $\mathbf{U}$ (a ) and components $V$

Like the regression example, we still have a linear model since PCA identifies a low-dimensional linear space

Both authors insist that their mode call PCA

For some reason I don't understand, people like to make a big

So we have seen that minimizing the sum of squared residuals (probably how)

The point of this post is to extend the idea of noise models to PCA and related models

<!-- ### Probabilistic PCA $\iff$ Istropic Gaussian noise model 

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

, but the notation is bloated and pretty much terrible in every other way you can think of. Even if notation was equal (its really not) the optimization viewpoint is conceptually much simpler in my view. Undergraduates in biology and social sciences are taught to "minimizing (squared) residuals" and not "maximizing likelihood under a Gaussian" -->


#### Other references:

* [StackExchange: PCA vs. Factor Analysis on the same dataset](http://stats.stackexchange.com/questions/94048/pca-and-exploratory-factor-analysis-on-the-same-dataset-differences-and-similar)


{% include sharebar.html %}

#### Footnotes
{:.no_toc}

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} Note that the likelihood treats $\mathbf{y}$ and $\mathbf{x}$ (vectors containing $x_i$ and $y_i$ for each observation) as constant variables.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} First, note that $\log(x)$ is a [monotonic function](https://en.wikipedia.org/wiki/Monotonic_function). If $x^{\*}$ is the value of $x$ that maximizes $f(x)$, then $x^{\*}$ also maximizes $g(f(x))$ for any monotonic function $g$. 
it is generally nicer to work with the [log-likelihood function](https://en.wikipedia.org/wiki/Likelihood_function#Log-likelihood) rather than directly with the likelihood function. This is perhaps especially true on computers, since the likelihood function involves the multiplication of many probabilities (numbers between 0 and 1) resulting in incredibly small numbers (think $10^{-n}$ for $n$ datapoints), which are difficult to represent on computers. All of these basic properties are good to know about. See more details [here](https://math.stackexchange.com/questions/892832/why-we-consider-log-likelihood-instead-of-likelihood-in-gaussian-distribution) and refer to g optimization and statistics textbooks, see e.g. https://web.stanford.edu/~boyd/cvxbook/.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=3 %} Roweis refers to probabilistic PCA as *sensible PCA* in his paper. He published concurrently with Tipping and Bishop, and their name has mostly won out.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=3 %} Applying $\log$ doesn't change the optimization problem because it is a monotonic function. Maximizing a function is equivalent to minimizing that function multiplied by $-1$.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=4 %} The matrix equation matches my last blog post, $ A = W C^T + R $. In this post $\mathbf{a}$ is a row in $A$, $\mathbf{w}$ is a row in $W$, and $\mathbf{c}$ is a column in $C$ (i.e. a row in $C^T$). The matrix $R \in \mathbb{R}^{n \times p}$ is the noise term: each $\boldsymbol{\xi}$ is a row in $R$.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=5 %} It's possible that there are more good explanation out there. Maybe I've just been unlucky with the papers I've read.
</p>
