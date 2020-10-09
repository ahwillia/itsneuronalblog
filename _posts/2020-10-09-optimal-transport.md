---
layout: post
title: A Short Introduction to Optimal Transport and Wasserstein Distance
comments: True
author: alex_williams
completed: True
topic: Optimal Transport
post_description: Some introductory notes on a recently popular topic in machine learning research.
---

<hr>

These notes provide a brief introduction to [optimal transport theory](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)), prioritizing intuition over mathematical rigor. Other good introductory resources include:

* Peyré & Cuturi (2019), ["**Computational Optimal Transport**"](http://dx.doi.org/10.1561/2200000073), Foundations and Trends® in Machine Learning: Vol. 11: No. 5-6, pp 355-607.
* Marco Cuturi's introductory lectures.
    * *MLSS Africa 2019.* &nbsp; [**Video Part I**](https://youtu.be/6iR1E6t1MMQ) -- [**Video Part II**](https://youtu.be/1ZiP_7kmIoc)
    * *MLSS Tübingen 2020.* &nbsp; [**Video Part I**](https://youtu.be/jgrkhZ8ovVc) -- [**Video Part II**](https://youtu.be/Bl8ZDN3Dbwk)


### Why Optimal Transport Theory?

A fundamental problem in statistics and machine learning is to come up with useful measures of "distance" between pairs of probability distributions.
Two desirable properties of a distance function are [symmetry](https://en.wikipedia.org/wiki/Symmetric_function) and the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality).
Unfortunately, many notions of "distance" between probability distributions do not satisfy these properties.
These weaker notions of distance are often called [*divergences*](https://en.wikipedia.org/wiki/Divergence_(statistics)).
Perhaps the most well-known divergence is the [Kullback-Lieibler (KL) divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence):

$$
\begin{equation}
D_{KL}(P \| Q) = \int p(x) \log \left ( \frac{p(x)}{q(x)} \right ) \mathrm{d}x
\end{equation}
$$

Where $P$ and $Q$ here denote probability distributions.
While the KL divergence is incredibly useful and fundamental in information theory, it also has its shortcomings.

For instance, one of the first things we learn about the KL divergence is that it is not symmetric, $D_{KL}(P \| Q) \neq D_{KL}(Q \| P)$.
This is arguably not a huge problem, since various symmetrized analogues to the KL divergence exist.
A bigger problem, in many cases, is that the divergence may be infinite if the [support](https://en.wikipedia.org/wiki/Support_(mathematics)) of $P$ and $Q$ are not equal.
Below we sketch three examples of 1D distributions for which $D_{KL}(P \| Q) = D_{KL}(Q \| P) = +\infty$.

{% include image.html url="/itsneuronalblog/code/ot/schematic_1d.png" width="500px" title="Three example distribution pairs, who are infinitely \"far apart\" according to KL divergence."%}

Intuitively, some of these distribution pairs seem "closer" to each other than others.
But the KL divergence says that they are all infinitely far apart.
One way of circumventing this is to smooth (i.e. add blur to) the distributions before computing the KL divergence, so that the support of $P$ and $Q$ matches.
However, choosing the bandwidth parameter of the smoothing kernel is not always straightforward.

Optimal transport theory helps us construct alternative notions of distance between probability distributions.
In particular, we will encounte the [Wasserstein distance](https://en.wikipedia.org/wiki/Wasserstein_metric) (also known as "Earth Mover's Distance" for reasons which will become apparent).
This distance is not only symmetric, but also satisfies the [triangle inequality](https://en.wikipedia.org/wiki/Triangle_inequality) (see, e.g., [Clement & Desch, 2008](https://doi.org/10.1090/S0002-9939-07-09020-X)).
Formally, if $\mathcal{W}(P, Q)$ denotes the Wasserstein distance between two distributions, then we have:

$$
\begin{equation}
\mathcal{W}(P, Q) = \mathcal{W}(Q, P) \quad\quad \text{and} \quad\quad \mathcal{W}(P, Q) \leq \mathcal{W}(P, M) + \mathcal{W}(M, Q)
\end{equation}
$$

for any probability distributions $P$, $Q$, and $M$.
Additionally, it can be shown $\mathcal{W}(P, Q) \neq 0$ unless $P = Q$.
These are all intuitively desirable properties to have in a measure of distance.
Indeed, distance functions that satisfy these properties are called [**metrics**](https://en.wikipedia.org/wiki/Metric_(mathematics)), and they play a foundational role in many areas of mathematics.

Interest in optimal transport seems to have markedly increased in recent years, with applications in imaging ([Lee et al., 2020](https://doi.org/10.1109/TCI.2020.3012954)), generative models ([Arjovsky et al., 2017](https://arxiv.org/abs/1701.07875)), and biological data analysis ([Schiebinger, 2019](https://doi.org/10.1016/j.cell.2019.01.006)), to name a few.

### An example transport problem in 2D

One of the nice aspects of optimal transport theory is that it can be grounded in physical intuition through the following thought experiment.
Suppose we are given the task of filling several holes in the ground.
The image below shows an overhead 2D view of this scenario &mdash; the three <span style="color:#D50000; font-weight:bold">red regions</span> correspond to dirt piles, and the eight <span style="color:#0000B6; font-weight:bold">blue regions</span> correspond to holes.

{% include image.html url="/itsneuronalblog/code/ot/holes.png" width="500px" title="Toy example in 2D." description="Overhead view of piled dirt (<span style="color:#D50000; font-weight:bold">red</span>) which must be transported to fill holes (<span style="color:#0000B6; font-weight:bold">blue</span>)."%}

Our goal is to come up with the *most efficient transportation plan* to which moves the dirt to fill all the holes.
We assume the total volume of the holes is equal to the total volume of the dirt piles.
In case it isn't clear where this is going &mdash; you should think of the piles as the probability density function of $P$ and the holes as the probability density function of $Q$.{%include footnote.html n=1 %}

The "most efficient" plan is the one that minimizes the total transportation cost.
To quantify this, let's say the **transportation cost** $C$ of moving 1 unit of dirt from $(x_0, y_0) \rightarrow (x_1, y_1)$ is given by the squared Euclidean distance:

$$
\begin{equation}
C(x_0, y_0, x_1, y_1) = (x_0 - x_1)^2 + (y_0 - y_1)^2
\end{equation}
$$

Other choices for the cost function are possible, but we will stick with this simple case.
Now we'll define the **transportation plan** $T$, which tells us how many units of dirt to move from $(x_0, y_0) \rightarrow (x_1, y_1)$.
For example, if the plan specifies:

$$
\begin{equation}
T(x_0, y_0, x_1, y_1) = w
\end{equation}
$$

then we intend to move $w$ units of dirt from position $(x_0, y_0) \rightarrow (x_1, y_1)$. For this to be a valid plan, start with at least $w$ units of dirt at $(x_0, y_0)$, and the depth of the hole at $(x_1, y_1)$ must be at least $w$ units. Also, we are only allowed to move positive units of dirt. We do allow dirt originating from $(x_0, y_0)$ to be split among multiple destinations.{%include footnote.html n=2 %}
In our 2D overhead view, we can visualize the transport $(x_0, y_0) \rightarrow (x_1, y_1)$ with an arrow like so:

{% include image.html url="/itsneuronalblog/code/ot/holes_arrow.png" width="500px" title="Example transport path." description="The arrow schematizes $w$ units of dirt being transported from location (x0, y0) to (x1, y1). A complete transport plan specifies transport paths like this over all pairs of locations."%}

The transportation plan, $T$, specifies an arrow like this from every possible starting position to every possible destination.
Further, in addition to being nonnegative $T(x_0, y_0, x_1, y_1) \geq 0$, the plan must satisfy the following two conditions:

$$
\begin{align}
\int \int T(x_0, y_0, x, y) \mathrm{d}x \mathrm{d}y &= p(x_0, y_0) \quad\quad \text{for all starting locations }(x_0, y_0).\\
\int \int T(x, y, x_1, y_1) \mathrm{d}x \mathrm{d}y &= q(x_1, y_1) \quad\quad \text{for all destinations }(x_1, y_1). \\
\end{align}
$$

Where $p(\cdot, \cdot)$ and $q(\cdot, \cdot)$ are density functions, which respectively correspond to the height of dirt and depth of hole.
The first equation says that the amount of piled dirt at $(x_0, y_0)$ is "used up" or transported somewhere.
The second equation says that the hole at $(x_1, y_1)$ is "filled up" with the required amount of dirt (no more, no less).

Suppose we are given a function $T$ that satisfies all of these conditions (i.e. we are given a *feasible* transport plan).{%include footnote.html n=3 %}
Then the overall transport cost is given by:

$$
\begin{equation}
\text{total cost} = \int \int \int \int C(x_0, y_0, x_1, y_1) \cdot T(x_0, y_0, x_1, y_1) \, \mathrm{d}x_0 \, \mathrm{d}y_0 \, \mathrm{d}x_1 \, \mathrm{d}y_1
\end{equation}
$$

This expression should be intuitive.
In essence, it states that that for every pointwise transportation $(x_0, y_0) \rightarrow (x_1, y_1)$ we multiply the amount of dirt transported, given by $T$, by the per unit transport cost, given by $C$.
Integrating over all possible origins and destinations gives us the total cost.

We've now fully formulated the optimal transport problem in 2D.
Taking a step back, here are a few questions and notes of interest about the problem:

* At first glance, finding the optimal transport plan $T$ might appear to be a really hard problem! However, we will show in the next section that, after discretizing the problem, finding the best transport plan amounts to solving a [linear program](https://en.wikipedia.org/wiki/Linear_programming). Perhaps easier than you might guess at first!


* We can interpret the transport plan as a probability distribution. Specifically, if $P$ and $Q$ are probability distributions over some space $\mathcal{X}$, then the transport plan can be viewed as a probability distribution over $\mathcal{X} \times \mathcal{X}$ where the operator "$\times$" denotes the [Cartesian product](https://en.wikipedia.org/wiki/Cartesian_product) (see also [*product measurable space*](https://en.wikipedia.org/wiki/Product_measure)). In our example above the space $\mathcal{X}$ corresponds to 2D Euclidean space, $\mathbb{R}^2$, and thus the transport plan a probability distribution on $\mathbb{R}^2 \times \mathbb{R}^2$ (which is isomorphic to 4D space, $\mathbb{R}^4$).


* The total transportation cost overcomes the two weaknesses of KL divergence we discussed at the beginning of this post. First, since the cost function $C$ is symmetric, the overall cost to transport $P \rightarrow Q$ is the same as transporting $Q \rightarrow P$. We will revisit this point again, but for now we schematize a simple 1D example to hopefully provide sufficient intuition for the symmetry:

{% include image.html url="/itsneuronalblog/code/ot/symmetry_1d.png" width="550px" title="Transport costs are symmetric." description="The density functions associated with $P$ and $Q$ are plotted on the left. In the middle and on the right we schematize the two possible transport problems. The symmetries in the problem (e.g. it is equally costly to move dirt left vs. right) mean that these two problems result in equivalent optimal transport costs."%}

* Recall the second shortcoming of KL divergence was that it was infinite for a variety of distributions with unequal support. Below we revisit the three simple 1D examples we showed at the beginning and compute the Wasserstein distance between them.{%include footnote.html n=4 %}
Not only is the Wasserstein distance finite in all cases, but the distances agree with our natural intuitions: the panel on the right results in the smallest Wasserstein distance, while the middle panel shows the largest distance.

{% include image.html url="/itsneuronalblog/code/ot/schematic_1d_revisited.png" width="550px" title="Unlike KL divergence, the Wasserstein distances in these examples are finite and intuitive."%}


### Solving the Optimization Problem

The optimization problem presented above is a [linear program](https://en.wikipedia.org/wiki/Linear_programming), so it can be solved in polynomial time by general-purpose algorithms. Here we'll use `scipy.optimize.linprog(...)` for demonstration purposes, but readers should note that there are more efficient and specialized algorithms (e.g., [Orlin's algorithm](https://doi.org/10.1287/opre.41.2.338); see also chapter 3 of [Peyré & Cuturi](http://dx.doi.org/10.1561/2200000073)). The function below computes the Wasserstein distance between two discrete distributions with probability mass functions `p` and `q` and with atoms located at `x`.

```python
def demo_wasserstein(x, p, q):
    """
    Computes order-2 Wasserstein distance between two
    discrete distributions.

    Parameters
    ----------
    x : ndarray, has shape (num_bins, dimension)
    
        Locations of discrete atoms (or "spatial bins")

    p : ndarray, has shape (num_bins,)

        Probability mass of the first distribution on each atom.

    q : ndarray, has shape (num_bins,)

        Probability mass of the second distribution on each atom.

    Returns
    -------
    dist : float

        The Wasserstein distance between the two distributions.

    T : ndarray, has shape (num_bins, num_bins)

        Optimal transport plan. Satisfies p == T.sum(axis=0)
        and q == T.sum(axis=1).

    Note
    ----
    This function is meant for demo purposes only and is not
    optimized for speed. It should still work reasonably well
    for moderately sized problems.
    """

    # Check inputs.
    if (abs(p.sum() - 1) > 1e-9) or (abs(p.sum() - q.sum()) > 1e-9):
        raise ValueError("Expected normalized probability masses.")

    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("Expected nonnegative mass vectors.")

    if (x.shape[0] != p.size) or (p.size != q.size):
        raise ValueError("Dimension mismatch.")

    # Compute pairwise costs between all xs.
    n, d = x.shape
    C = squareform(pdist(x, metric="sqeuclidean"))

    # Scipy's linear programming solver will accept the problem in
    # the following form:
    # 
    # minimize     c @ t        over t
    # subject to   A @ t == b
    #
    # where we specify the vectors c, b and the matrix A as parameters.

    # Construct matrices Ap and Aq encoding marginal constraints.
    # We want (Ap @ t == p) and (Aq @ t == q).
    Ap, Aq = [], []
    z = np.zeros((n, n))
    z[:, 0] = 1

    for i in range(n):
        Ap.append(z.ravel())
        Aq.append(z.transpose().ravel())
        z = np.roll(z, 1, axis=1)

    # We can leave off the final constraint, as it is redundant.
    # See Remark 3.1 in Peyre & Cuturi (2019).
    A = np.row_stack((Ap, Aq))[:-1]
    b = np.concatenate((p, q))[:-1]

    # Solve linear program, recover optimal vector t.
    result = linprog(C.ravel(), A_eq=A, b_eq=b)

    # Reshape optimal vector into (n x n) transport plan matrix T.
    T = result.x.reshape((n, n))

    # Return Wasserstein distance and transport plan.
    return np.sqrt(np.sum(T * C)), T
```

### An Example in 1D

To demonstrate what this looks like, let's first consider a 1D example.
On the left panel below we show two probability mass functions defined on the interval $[0, 1]$.
On the right, visualize the cost matrix $\mathbf{C}$ along with the same density functions (one up top and other flipped vertically).
The cost is zero along the diagonal of $\mathbf{C}$ since it costs us nothing to move mass zero units of distance.
Since we define the transportation cost as squared Euclidean distance, moving vertically or horizontally off the diagonal increases the cost like $x^2$.

<img src="/itsneuronalblog/code/ot/example_1d_transport_cost.png" width=550>

The figure above displays all the necessary ingredients for us to find the optimal transport plan: two target marginal distributions $\mathbf{p}$ and $\mathbf{q}$ and the cost matrix $\mathbf{C}$. We input these three ingredients into our the linear programming solver and are given back the optimal transport plan $\mathbf{T}^\*$.
This transport plan is a matrix the same size as $\mathbf{C}$ and is shown below on the right:

<img src="/itsneuronalblog/code/ot/example_1d_transport_plan.png" width=550>

By inspecting this transport plan, we can appreciate a few high-level patterns.
First, $\mathbf{T}^\*$ is very sparse, and nonzero entries trace out a curved path from the upper right to the lower left corner.
This is intuitive &mdash; the mass two nearby locations, $x$ and $x + \delta x$, has a similar transport cost to all locations, so we would expect their destination to be similar (especially because the marginal densities are smooth in this example).

Second, the largest peaks in $\mathbf{T}^\*$ (the parts colored yellow) correspond to peaks in the marginal densities.
Conversely, dark spots in the transport plan correspond to troughs in $\mathbf{p}$ and $\mathbf{q}$.
This is also intuitive because the transport plan is constrained to match these marginal distributions; expressed in Python, we have `T.sum(axis=0) == p` and `T.sum(axis=1) == q` (up to floating point precision).
Finally, the nonzero elements in $\mathbf{T}^\*$ lie below the diagonal.
This is because most of the mass in $\mathbf{p}$ is to the left of the mass in $\mathbf{q}$.

### An Example in 2D

Now let's return to our original 2D problem and see what the solution looks like in this more complex setting.
As mentioned before, we will discretize our problem into spatial bins.
Here, I've chosen a 20 x 20 grid, which is rather coarse-grained, but it works for demonstration purposes.
The left panels shows the discretized 2D heat maps &mdash; $\mathbf{p}$ corresponds to the three dirt piles, and $\mathbf{q}$ corresponds to the scattered holes.
On the right, we plot these same densities after flattening these 2D densities into 1D vectors, and also plot the cost matrix, $\mathbf{C}$.
Since we have a 20 x 20 discrete grid, there are a total of 400 bins, and thus $\mathbf{C}$ is a 400 x 400 matrix.

<img src="/itsneuronalblog/code/ot/discretized_holes_cost.png" width=550>

Because these visualizations reduce the 2D distributions down to a single dimension, they are a bit more complicated and tricky to interpret than the 1D case.
Here I linearized the 2D grid of bins by the standard `numpy.ravel()`, so after a bit of reflection the blocky structure of the cost matrix above should make since.
Rather than getting lost in these details, the important point is that we have reduced the 2D problem to something similar to the 1D example we considered in the last section, and we can use the same code to identify the optimal transport plan, $\mathbf{T}^\*$.
Doing this, we obtain the following:

<img src="/itsneuronalblog/code/ot/discretized_holes_transport.png" width=550>

It is pretty difficult to visually interpret this optimal transport plan as it is extremely sparse &mdash; in fact, I had to add a little bit of Gaussian blur to the heatmap so that the yellow spots, corresponding to peaks in $\mathbf{T}^\*$, are visible.
Regardless, it is very satisfying that the same linear programming approach worked for us as in the 1D example above.
If we wanted to, we could now take the inner product between $\mathbf{T}^\*$ and $\mathbf{C}$ and then take the square root to arrive at the Wasserstein distance between $P$ and $Q$.

Though this is enough to demonstrate the basic idea, it would be a bit dissatisfying to end without something a little more intuitive.
Below, I took the largest 80 entries of the optimal transport plan, which is plotted above as a heatmap.
Each of these entries, $\mathbf{T}^{\*}\_{ij}$, specifies an origin (bin $i$) and a destination (bin $j$).
When we overlay these 80 arrows on top of our (discretized) 2D densities, we get very intuitive and satisfying result:

<img src="/itsneuronalblog/code/ot/holes_arrows.png" width=450>


### Entropy Regularization

Before concluding, I want to quickly mention an important innovation that has galvanized recent work on optimal transport in machine learning.
Very briefly, the idea is to penalize transport plans with small [Shannon entropy](https://en.wikipedia.org/wiki/Entropy_(information_theory)).
To do this we modify the optimization problem as follows:

$$
\begin{align}
&\underset{\mathbf{T}}{\text{minimize}} & & \langle \mathbf{T}, \mathbf{C} \rangle - \epsilon H(\mathbf{T}) \\
&\text{subject to} & & \mathbf{T} \boldsymbol{1} = \mathbf{p},~~ \mathbf{T}^\top \boldsymbol{1} = \mathbf{q}, ~~ \mathbf{T} \geq 0
\end{align}
$$

Here, $\epsilon > 0$ is the strength of the regularization penalty and $H(\mathbf{C}) = -\sum\_{ij} \mathbf{T}\_{ij} \log \mathbf{T}\_{ij}$ is the Shannon entropy.<sup>**[7]**</sup>
As $\epsilon \rightarrow 0$, we of course cover our original optimal transport problem.
As $\epsilon \rightarrow \infty$ it can be shown that the optimal transport plan is given by $\mathbf{T}\_{ij}^\* = \mathbf{p}\_i \mathbf{q}\_j$, so intuitively the problem becomes progressively easier to solve as we increase $\epsilon$.
You can think of the regularization term as reducing sparsity in optimal transport plan and discouraging the solution from hiding out in the sharp edges of the [polytope](https://en.wikipedia.org/wiki/Convex_polytope) defined by the linear constraints of the problem.

The figure below (reproduced from Peyré & Cuturi, Fig. 4.2) shows the effect of decreasing the regularization strength for a simple 1D optimal transport problem.
The marginal densities are shown by the blue and red lines on the bottom panels.
The colored heatmaps (top) and 2d surface plots (bottom) visualize the optimal transport plan for various values of $\epsilon$.

<img src="/itsneuronalblog/code/ot/entropic_regularization.png" width=550>

The computational advantages of entropy regularization are substantial for high-dimensional data.
If we discretize the space into $d$ bins (as we did in the previous section) then we can expect the computational expense to be $O(d^3 \log d)$.<sup>**[8]**</sup> 
In contrast, we can expect *nearly linear time* convergence after adding the entropy regularization, as established by recent work ([Altschuler et al., 2019](https://arxiv.org/abs/1705.09634) ; [Dvurechensky et al., 2019](https://arxiv.org/abs/1802.04367)).
Chapter 4 of [Peyré & Cuturi (2019)](http://dx.doi.org/10.1561/2200000073) provides a good introduction for the algorithmic tricks and interpretations of this entropy-regularized problem.

### Footnotes

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} Or vice versa! It shouldn't be hard to see that the problem is entirely symmetric &mdash; it would cost us the same to transport the dirt back out of the holes as it did to transport the dirt there in the first place, so we can think about transport in either direction as being equivalent.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Allowing dirt to be split in this fashion corresponds to the [Kanotorovich formulation](https://en.wikipedia.org/wiki/Transportation_theory_(mathematics)#Monge_and_Kantorovich_formulations) of the transport problem, which is distinct from the original formulation which dates back to [Gaspard Monge](https://en.wikipedia.org/wiki/Gaspard_Monge). We stick to Kanotorovich's formulation because it is more analytically and computationally tractable (and thus more common in modern applications).
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=3 %} One might wonder &mdash; does a feasible transport plan always exist? Yes! One can check that the [*product measure*](https://en.wikipedia.org/wiki/Product_measure), $T(x_0, y_0, x, y) = p(x_0, y_0) q(x_1, y_1)$, satisfies all the required constraints.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=4 %} To compute the Wasserstein distance in each of these cases, we used the [`scipy.stats.wasserstein_distance`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html) function, which is super handy. Unfortunately it only works for 1D distributions.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=5 %} There are two notable cases where optimal transport plans can be computed analytically. We state these cases briefly here; further details and references can be found in [(Peyré & Cuturi, 2019; Remarks 2.30 and 2.31)](http://dx.doi.org/10.1561/2200000073).
<br><br>
**Univariate distributions.** Let $f^{-1}(\cdot)$ and $g^{-1}(\cdot)$ denote the [inverse c.d.f.s](https://en.wikipedia.org/wiki/Quantile_function) of two univariate distributions.
Then, the order-$p$ Wasserstein distance between the distributions is given by $(\int_0^1 |f^{-1}(y) - g^{-1}(y)|^p dy )^{1/p}$.
<br>
**Gausssian Distributions.** Given two normal distributions with means $(\mu_1, \mu_2)$ and covariances $(\Sigma_1, \Sigma_2)$, then the (second order) Wasserstein distance between the distributions is:
$$
(\Vert \mu_1 - \mu_2 \Vert_2^2 + \mathcal{B}(\Sigma_1, \Sigma_2)^2)^{1/2}
$$
where $\mathcal{B}$ denotes the [*Bures metric*](https://arxiv.org/abs/2001.08056) on positive-definite matrices.
For univariate normal distributions this simplifies to:
$$
((\mu_1 - \mu_2)^2 + (\sigma_1 - \sigma_2)^2)^{1/2}
$$
where $\sigma_1$ and $\sigma_2$ denote the standard deviations.
That is, the Wasserstein distance between two 1D gaussians is equal to the Euclidean distance of the parameters plotted in the 2D plane, with axes corresponding to the mean and standard deviation.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=6 %} Here, we've defined the Wasserstein distance for two discrete distributions, but it can also be defined (though not easily computed) for continuous distributions. See, e.g., the formal definition of [Wasserstein distance on Wikipedia](https://en.wikipedia.org/wiki/Wasserstein_metric). Further, this post only covers the "2nd order" Wasserstein distance for simplicity. More generally, if we define the per-unit costs as $\mathbf{C}_{ij} = \Vert \mathbf{x}_i - \mathbf{x}_j \Vert^p_2$ then the Wasserstein distance of order $p$ is given by $\langle \mathbf{T}^\*, \mathbf{C} \rangle^{1/p}$. Order $p=1$ Wasserstein distance is also of practical interest since it tends to be more robust to outliers. See chapter 6 of [Peyré & Cuturi (2019)](http://dx.doi.org/10.1561/2200000073) for further discussion.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=7 %} Note that [Peyré & Cuturi (2019)](http://dx.doi.org/10.1561/2200000073) define the entropy term slightly differently as $H(\mathbf{T}) = -\sum\_{ij} \mathbf{T}\_{ij} \log \mathbf{T}\_{ij} + \sum\_{ij} \mathbf{T}\_{ij}$, but the constraints of our problem imply that $\sum\_{ij} \mathbf{T}\_{ij} = 1$ so the only difference is an additive constant. These discrepancies do become important in other cases, such as in the case of unbalanced optimal transport (see section 10.2 of Peyré & Cuturi, 2019).
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=8 %} This is the computational complexity of [Orlin's algorithm](https://doi.org/10.1287/opre.41.2.338) which appears to be the current state-of-the-art based on the discussion in [Altschuler et al. 2019](https://arxiv.org/abs/1705.09634).
</p>