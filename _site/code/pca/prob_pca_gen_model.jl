using PyPlot
close("all")

n = 200 # number of observations
p = 3   # number of features
r = 2   # subspace dimension

# loadings and components
w = randn(n,r) 
C = 5*[ 1.0  -0.5
        1.0   0.0
        1.0   1.0 ]

a = w*transpose(C) + randn(n,p)

figure()
subplot(2,2,1)
plot(w[:,1],w[:,2])
subplot(2,2,2)
plot3D(a[:,1],a[:,2],a[:,3])
