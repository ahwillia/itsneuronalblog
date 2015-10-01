---
layout: post
title: What is clustering and why is it hard?
comments: True
author: alex_williams
completed: True
topic: Clustering
---

I've been working on some clustering techniques to [identify cell types from DNA methylation data](http://alexhwilliams.info/pubs/DoE_2015_DNA_meth.compressed.pdf). When you dive into the literature on clustering, two things becomes immediately apparent: first, clustering is fundamental to many scientific questions, and second, there is ["distressingly little general theory"](http://stanford.edu/~rezab/papers/slunique.pdf) on how it works or how to apply it to your particular data.

This was surprising to me. I imagine that most biologists and neuroscientists come across [k-means clustering](https://en.wikipedia.org/wiki/K-means_clustering), [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), and similar techniques all the time in papers related to their work. Given how commonplace these techniques are, one would think that we have a solid handle on how they work and what can go wrong.

This will be the first post in a short series on clustering techniques. I will try to explain why clustering is hard from a high-level, intuitive perspective. The next post will cover some more technical theoretical results. I'll focus on [Jon Kleinberg's paper](http://web.stanford.edu/~rezab/classes/cme305/W15/Notes/Kleinberg%20-%20impossibility%20theorem.pdf) which precisely defines an ideal clustering function, but then proves that ***no such function exists*** and that there are inevitable tradeoffs that must be made. The final few posts will cover other theoretical work and some current projects of mine.

<!--more-->

### What is clustering?

[*Clustering*](https://en.wikipedia.org/wiki/Clustering) is difficult because it is an *unsupervised learning* problem: we are given a dataset and are asked to infer structure within it (in this case, the latent clusters/categories in the data). The problem is that there isn't necessarily a "correct" or ground truth solution that we can refer to if we want to check our answers. This is in contrast to [*classification problems*](https://en.wikipedia.org/wiki/Statistical_classification), where we do know the ground truth. Deep artificial neural networks are very good at classification ([*NYT article*](http://bits.blogs.nytimes.com/2014/08/18/computer-eyesight-gets-a-lot-more-accurate/); [Deng et al. 2009](http://www.image-net.org/papers/imagenet_cvpr09.pdf)), but clustering is still a very open problem.

For example, it is a classification problem to predict whether or not a patient has a common disease based on a list of symptoms. In this case, we can draw upon past clinical records to make this judgment, and we can gather further data (e.g. a blood test) to confirm our prediction. In other words, we assume there is a self-evident ground truth (the patient either has or does not have disease X) that can be observed. 

For clustering, we lack this critical information. For example, suppose you are given a large number of beetles and told to group them into clusters based on their appearance. Assuming that you aren't an entomologist, this will involve some judgment calls and guesswork.<sup>[1]</sup> If you and a friend sort the same 100 beetles into 5 groups, you will likely come up with slightly different answers. And &mdash; here's the important part &mdash; there isn't really a way to determine which one of you is "right".

### Approaches for clustering

There is a lot of material written on this already, so rather than re-hash what's out there I will just point you to the best resources.

* **K-means clustering**  ([*Wikipedia*](https://en.wikipedia.org/wiki/K-means_clustering), [Visualization by @ChrisPolis](http://www.bytemuse.com/post/k-means-clustering-visualization/), [Visualization by TECH-NI blog](http://tech.nitoyon.com/en/blog/2013/11/07/k-means/)).

* **Hierarchical clustering** works by starting with each datapoint in its own cluster and fusing the nearest clusters together repeatedly ([*Wikipedia*](https://en.wikipedia.org/wiki/Hierarchical_clustering), [*Youtube #1*](https://youtu.be/XJ3194AmH40), [*Youtube #2*](https://youtu.be/VMyXc3SiEqs)).

	* [**Single-linkage clustering**](https://en.wikipedia.org/wiki/Single-linkage_clustering) is a particularly popular and well-characterized form of hierarchical clustering. Briefly, single-linkage begins by initializing each point as its own cluster, and then repeatedly combining the two closest clusters (as measured by their closest points of approach) until the desired number of clusters is achieved.

* **Bayesian methods** include [finite mixture models](http://ifas.jku.at/gruen/BayesMix/bayesmix-intro.pdf) and [infinite mixture models](http://www.kyb.tue.mpg.de/fileadmin/user_upload/files/publications/pdfs/pdf2299.pdf).<sup>[2]</sup>

The important thing to realize is that all of these approaches are [very computationally difficult](http://cseweb.ucsd.edu/~avattani/papers/kmeans_hardness.pdf) to solve exactly for large datasets (more on this in my next post). As a result, we often resort to optimization heuristics that may or may not produce reasonable results. And, as we will soon see, even if the results are "reasonable" from the algorithm's perspective, they might not align with our intuition, prior knowledge, or desired outcome.

### It is difficult to determine the number of clusters in a dataset

This has to be the most widely understood problem with clustering. In fact, there is an [*entire Wikipedia article*](https://en.wikipedia.org/wiki/Determining_the_number_of_clusters_in_a_data_set) devoted to it. If you think about the problem for long enough, you will come to the inescapable conclusion is that there is no "true" number of clusters (though some numbers ***feel*** better than others), and that the same dataset is appropriately viewed at various levels of granularity depending on analysis goals.

{% include image.html url="/itsneuronalblog/img/clustering/clustering_ambiguity.png" width="400px" title="" description=""%}

While this problem cannot be "solved" definitively, there are some nice ways of dealing with it. Hierarchical clustering approaches provide cluster assignments for all possible number of clusters, allowing the analyst or reader to view the data across different levels of granularity. There are also Bayesian approaches such as [Dirichlet Process Mixture Models](http://blog.echen.me/2012/03/20/infinite-mixture-models-with-nonparametric-bayes-and-the-dirichlet-process/) that adaptively estimate the number of clusters based on a hyperparameter which tunes dispersion. A number of recent papers have focused on *convex clustering* techniques that fuse cluster centroids together in a continuous manner along a regularization path; this exposes a hierarchical structure for a clustering approach (roughly) similar to k-means.<sup>[3]</sup> Of course, there are many other papers out there on this subject.<sup>[4]</sup> 

### It is difficult to cluster outliers (even if they form a common group)

I recommend you read David Robinson's excellent post on [the shortcomings of k-means clustering](http://stats.stackexchange.com/questions/133656/how-to-understand-the-drawbacks-of-k-means). The following example he provides is particularly compelling:

{% include image.html url="/itsneuronalblog/img/clustering/clustering_01.png" width="400px" title="Raw Data." description="Three spherical clusters with variable numbers of elements/points."%}

The human eye can pretty easily separate these data into three groups, but the k-means algorithm fails pretty hard:

{% include image.html url="/itsneuronalblog/img/clustering/clustering_02.png" width="500px" title="" description=""%}

Rather than assigning the points in the upper left corner to their own cluster, the algorithm breaks the largest cluster (in the upper right) into two clusters. In other words it tolerates a few large errors (upper left) in order to decrease the errors where data is particularly dense (upper right). This likely doesn't align with our analysis, but it is *completely reasonable* from the perspective of the algorithm. And again, ***there isn't a ground truth to show that the algorithm is "wrong" per se.***

### It is difficult to cluster non-spherical, overlapping data

A final, related problem arises from the shape of the data clusters. Every clustering algorithm makes structural assumptions about the dataset that need to be considered. For example, k-means works by minimizing the total sum-of-squared distance to the cluster centroids. This can produce undesirable results when the clusters are elongated in certain directions &mdash; particularly when the between-cluster distance is smaller than the maximum within-cluster distance. Single-linkage clustering, in contrast, can perform well in these cases, since points are clustered together based on their nearest neighbor, which facilitates clustering along 'paths' in the dataset.

{% include image.html url="/itsneuronalblog/img/clustering/kmeans_fail.png" width="500px" title="Datasets where single-linkage outperforms k-means." description="If your dataset contains long paths, then single-linkage clustering (panels B and D) will typically perform better than k-means (panels A and C)."%}

However, there is [no free lunch](https://en.wikipedia.org/wiki/No_free_lunch_theorem).<sup>[5]</sup> Single-linkage clustering is more sensitive to noise, because each clustering assignment is based on a single pair of datapoints (the pair with minimal distance). This can cause paths to form between overlapping clouds of points. In contrast, k-means uses a more global calculation &mdash; minimizing the distance to the nearest centroid summed over all points. As a result, k-means typically does a better job of identifying partially overlapping clusters.

{% include image.html url="/itsneuronalblog/img/clustering/linkage_fail.png" width="550px" title="A dataset where k-means outperforms single-linkage." description="Single-linkage clustering tends to erroneously fuse together overlapping groups of points (red dots); small groups of outliers (black dots) are clustered together based on their small pairwise distances."%}

The above figures were schematically reproduced from [these lecture notes](http://www.stat.cmu.edu/~cshalizi/350/lectures/08/lecture-08.pdf) from a statistics course at Carnegie Mellon.

### Are there solutions these problems?

This post was meant to highlight the inherent difficulty of clustering rather than propose solutions to these issues. It may therefore come off as a bit pessimistic. There are many heuristics that can help overcome the above issues, but I think it is important to emphasize that these are ***only heuristics***, not guarantees. While many of biologists treat k-means as an "off the shelf" clustering algorithm, we need to be at least a little careful when we do this.

One of the more interesting heuristics worth reading up on is called [ensemble clustering](http://dx.doi.org/10.1109/ICPR.2002.1047450). The basic idea is to average the outcomes of several clustering techniques or from the same technique fit from different random initializations. Each clustering fit may suffer from instability, but the average behavior of the ensemble of models will tend to be more desirable. This general trick is called [*ensemble averaging*](https://en.wikipedia.org/wiki/Ensemble_averaging) and has been successfully applied to a number of machine learning problems.<sup>[6]</sup>

This post only provides a quick outline of the typical issues that arise for clustering problems. The details of the algorithms have been purposefully omitted, although a deep understanding of these issues likely requires a closer look at these specifics. [Jain (2010)](http://www.cse.msu.edu/biometrics/Publications/Clustering/JainClustering_PRL10.pdf) provides a more comprehensive review that is worth reading.

{% include sharebar.html %}

#### Footnotes

<span class="footnotes">
**[1]** It will probably involve guesswork even if you are.<br>
**[2]** Highlighting the difficulty of clustering, [Larry Wasserman](https://normaldeviate.wordpress.com/2012/08/04/mixture-models-the-twilight-zone-of-statistics/) has joked that "mixtures, like tequila, are inherently evil and should be avoided at all costs." [Andrew Gelman](http://andrewgelman.com/2012/08/15/how-to-think-about-mixture-models/) is slightly less pessimistic.<br>
**[3]** Convex clustering performs continuous clustering, similar to how [LASSO](http://statweb.stanford.edu/~tibs/lasso/lasso.pdf) performs continuous variable selection.<br>
**[4]** See [Tibshirani et al. (2001)](http://dx.doi.org/10.1111/1467-9868.00293), [Dudoit & Fridlyand (2002)](http://dx.doi.org/10.1186/gb-2002-3-7-research0036), [Figueiredo & Jain (2002)](http://dx.doi.org/10.1109/34.990138), [Yan & Ye (2007)](http://dx.doi.org/10.1111/j.1541-0420.2007.00784.x), [Kulis & Jordan (2012)](http://www.cs.berkeley.edu/~jordan/papers/kulis-jordan-icml12.pdf)<br>
**[5]** The ["no free lunch" theorem](http://ti.arc.nasa.gov/m/profile/dhw/papers/78.pdf) roughly states that whenever an algorithm performs well on a certain class of problems it is because it makes *good assumptions* about those problems; however, you can always construct new problems that violate these assumptions, leading to worse performance. Interestingly, this basic idea pops up in other contexts. For example, certain feedback control systems can be engineered so that they are robust to particular perturbations, but such engineering renders them ***more sensitive*** to other forms of perturbations (see ["waterbed effect."](https://en.wikipedia.org/wiki/Bode%27s_sensitivity_integral))<br>
**[6]** See [*bootstrap aggregation*](https://en.wikipedia.org/wiki/Bootstrap_aggregating), [*random forests*](https://en.wikipedia.org/wiki/Random_forest), and [*ensemble learning*](https://en.wikipedia.org/wiki/Ensemble_learning). Seminal work on this topic was done by [Leo Breiman ](https://en.wikipedia.org/wiki/Leo_Breiman) &mdash; his papers are lucid, fascinating, and accessible, and I particularly recommend his 1996 article ["Bagging Predictors"](http://dx.doi.org/10.1007/BF00058655). This is also covered in any modern textbook, such as *The Elements of Statistical Learning*.
</span>