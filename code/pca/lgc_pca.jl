#Pkg.add("Distributions")
#Pkg.add("LowRankModels")
#Pkg.add("PyPlot")
using Distributions
using LowRankModels
using PyPlot

m,n = 300,50
m1,m2,m3 = 1:100,101:200,201:300

lgc(x) = 1/(1+exp(-x))

function sample_cluster(m)
	μ = randn(n)
	Σ = randn(n,1)*randn(1,n)+randn(n,n)/10
	Σ = Σ*Σ'
	return rand(MvNormal(μ,Σ),m)'
end

A1 = sample_cluster(length(m1))
A2 = sample_cluster(length(m2))
A3 = sample_cluster(length(m3))

A = [A1;A2;A3]
Ad = zeros(m,n)

for i = 1:m
	for j = 1:n
		if rand() < lgc(A[i,j])
			Ad[i,j] = 1
		else
			Ad[i,j] = -1
		end
	end
end

g1 = GLRM(Ad,LogisticLoss(),ZeroReg(),ZeroReg(),2)
g2 = GLRM(Ad,QuadLoss(),ZeroReg(),ZeroReg(),2)

fit!(g1),fit!(g2)

# figure()
# imshow(Ad,interpolation="None",cmap=ColorMap("bwr"))

figure()
subplot(1,2,1)
title("logistic PCA")
plot(g1.X[1,m1],g1.X[2,m1],".b")
plot(g1.X[1,m2],g1.X[2,m2],".r")
plot(g1.X[1,m3],g1.X[2,m3],".g")
subplot(1,2,2)
title("classic PCA")
plot(g2.X[1,m1],g2.X[2,m2],".b")
plot(g2.X[1,m2],g2.X[2,m2],".r")
plot(g2.X[1,m3],g2.X[2,m3],".g")
show()
