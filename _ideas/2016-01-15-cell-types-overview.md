---
layout: post
title: How many different cell types are in the brain? And is this a well-posed and useful question?
comments: True
author: alex_williams
topic: Cell Types
---

In early 2014, the BRAIN initiative identified [*"Transformative Approaches for Cell-Type Classification in the Brain"*](http://grants.nih.gov/grants/guide/rfa-files/RFA-MH-14-215.html) as a fundamental research goal for the next decade. Since then, the literature has exploded with [*paper*](http://dx.doi.org/10.1126/science.aac9462)... after [*paper*](http://dx.doi.org/10.1038/nature16468), after [*paper*](http://dx.doi.org/10.1038/nbt.3443)... after [*paper*](http://dx.doi.org/10.1038/nbt.3445)... after [*giant Allen Institute database release*](http://celltypes.brain-map.org/)... after [*paper*](http://dx.doi.org/10.1126/science.aaa1934)... after [*paper*](http://dx.doi.org/10.1152/jn.00237.2015) on cell types.

I'm attracted to this line of research for a few reasons. First, I spent my formative undergraduate years studying the crustacean stomatogastric ganglion and related simple invertebrate circuits. The major selling point for studying these systems is that their cell types and their connectivity were well-characterized long ago. This enables very different kinds of scientific inquiry, particularly with regard to studying the [variability](#) and [stability](#) of circuit parameters. Once you can attach definitive labels to all the cells in a circuit, you can ask how those cells 

I also think the computational and statistical challenges in defining cell types in mammalian cortex are interesting in their own right. There are a staggering number of neurons (let alone glia) in the brain, but there are certainly redundancies, symmetries, and other simple relationships that we can extract to make our job at least a little bit easier. Finding the underlying simplicity in a sea of complex data lies at the heart of many interesting questions in modern science and engineering.

Despite these reasons for excitement, I'm not sure I'd know what to do if you handed me a detailed atlas of cell types in each region of the brain. How would I know if I should trust such an atlas? Might I be misled by viewing cells as discrete types, instead of, say, a lying on a continuous spectrum?

I get a little nervous when there is such intense focus and rapid advancement on a particular topic. Is the hype justified? This is my brief attempt to summarize what's been done and shed light on this question.

<!--more-->

### Basic features to distinguish different cell types

#### Morphology

#### Electrophysiology

#### mRNA expression

#### Epigenetic markers

### Statistical and computational challenges for defining cell types

First and foremost, it is important to remind ourselves that our problems as neuroscientists are not special or singular. The problem of defining cell types is really just a clustering problem; it belongs to a larger family of statistical problems encountered in many different disciplines for many decades. And what we've learned from our collective experience is that [clustering](#) [is hard](#) ([*usually*](#)). 

#### Determining the number of cell types (i.e. clusters)

* http://stats.stackexchange.com/questions/23472/how-to-decide-on-the-correct-number-of-clusters
* https://www.stat.washington.edu/raftery/Research/PDF/fraley1998.pdf

#### Large numbers of possible features

#### Noisy, zero-inflated data

#### Combining disparate data types into a unified picture

### Is this useful?

<blockquote class="twitter-tweet tw-align-center" lang="en"><p lang="en" dir="ltr"><a href="https://twitter.com/neuroecology">@neuroecology</a> <a href="https://twitter.com/neurofim">@neurofim</a> stamp collecting</p>&mdash; Justin Kiggins (@neuromusic) <a href="https://twitter.com/neuromusic/status/683745779509403649">January 3, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

<blockquote class="twitter-tweet tw-align-center" lang="en"><p lang="en" dir="ltr"><a href="https://twitter.com/neuroecology">@neuroecology</a> <a href="https://twitter.com/neurofim">@neurofim</a> computational stamp collecting</p>&mdash; Justin Kiggins (@neuromusic) <a href="https://twitter.com/neuromusic/status/683888384578719744">January 4, 2016</a></blockquote>
<script async src="//platform.twitter.com/widgets.js" charset="utf-8"></script>

I mentioned at the beginning of this post that 

The other important feature of simple invertebrate circuits is that they control simple, well-defined, and easily quantified behaviors. Let's say we zoom in and characterize all of the cell types in mouse visual cortex. It would be natural to then test how manipulating the activity/number/connectivity of various cell types affects simple measures of early sensory processing (i.e. receptive fields).

The results may not be too surprising... Manipulating the activity of a cell type will change receptive field properties. If you believe that all the cells in V1 are solely there to produce receptive fields then this may be a fine approach. The nagging worry, for me, is that maybe certain cell types are concerned with doing other things, but we could easily fool ourselves into thinking they control feature x, y, or z of visual receptive fields.

#### Short Bibliography

{% include sharebar.html %}

#### Footnotes

<p class="footnotes" markdown="1">
{% include foot_bottom.html n=1 %} Shaping the activity-dependent development of the circuit, stabilizing circuit activity in the absence of visual input, etc.
</p>
<p class="footnotes" markdown="1">
{% include foot_bottom.html n=2 %} Footnote 2.
</p>
