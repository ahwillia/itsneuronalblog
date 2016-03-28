using PyPlot
close("all")

c_true = [1.0, 2.0, 3.0, 3.0, -1.5, -1.0, -0.5, 1.0]
w_true = [1.0, 2.0, 2.0, 1.0, 0.5, 1.0,-1.0, -2.0, -0.5, -1.0, -1.0, -2.0]

A = w_true*c_true'
A += randn(size(A))*0.6
for i = 1:size(A,2)
	A[:,i] -= mean(A[:,i])
end
cl = maximum(abs(A))

subplot(221)
imshow(A,cmap=ColorMap("bwr"),interpolation="none")
title("data")
clim([-cl,cl])
xticks([]), yticks([])

U,s,Vt = svd(A)
w = s[1]*U[:,1:1]
c = Vt[:,1:1]

subplot(222)
imshow(c',cmap=ColorMap("bwr"),interpolation="none")
title("top principal component, c")
clim([-maximum(abs(c)),maximum(abs(c))])
xticks([]), yticks([])

subplot(223)
imshow(w,cmap=ColorMap("bwr"),interpolation="none")
title("loadings, w")
clim([-maximum(abs(w)),maximum(abs(w))])
xticks([]), yticks([])

subplot(224)
imshow(w*c',cmap=ColorMap("bwr"),interpolation="none")
title("reconstructed data")
clim([-cl,cl])
xticks([]), yticks([])

savefig("ex1.png")
