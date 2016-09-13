------------------ Notes from meeting with Jeff ------------------------

* "The tensor cookbook"?

* Basic introduction to tensor operations.
    * Matrix Computations, last chapter
    
* https://www.youtube.com/watch?v=tpL95Sd7zT0

* The ubiquitous Kronecker product

* Moitra's video
















What are some examples of these useful extensions? Sometimes we have data that is quite naturally expressed as a 3D table rather than a matrix. Suppose we measure mRNA expression across the development of an animal. Then at each developmental timepoint we have a $n \times p$ matrix of data as described above. These matrices can be stacked into a 3D data structure, which we call a ***tensor*** (specifically, in this case, a third-order tensor). A number of questions arise: Can we apply PCA to this 3D table of data? Are there advantages of expressing the data as a tensor, rather than a matrix? This is the flavor of topics I want to blog about in the next few posts.

{% include image.html url="/itsneuronalblog/img/pca/matrix-01.png" width="800px" title="Matrix vs Tensor frameworks for representing data." %}

This sets the stage for the next post on [**CP Tensor Decomposition**](https://en.wikipedia.org/wiki/Tensor_rank_decomposition) (which is one way of generalizing PCA to tensors):

* Represent the data as a 3D (or higher!) array of data $A$
* We define a cost function / regularization
* We optimize three small matrices whose [tensor product](https://en.wikipedia.org/wiki/Outer_product#Tensor_multiplication) reconstructs the data as best as possible.
* The optimization problem is triconvex. We can still use [sequential convex programming](http://stanford.edu/class/ee364b/lectures/seq_slides.pdf) methods!




