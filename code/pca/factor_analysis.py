import numpy as np
import pylab as plt
from sklearn.decomposition import FactorAnalysis

def main(n=100,p=50,r=5):
	
	# number of replicates
	nr = 15

	# vary noise levels over wide range
	nl = 10
	noise_lev = np.zeros(nl)
	noise_lev[1:] = np.logspace(-1,1,nl-1)
	
	# compare pca to factor analysis over noise levels
	pe,fe,nv = np.empty((nl,nr)),np.empty((nl,nr)),np.empty((nl,nr))
	for j in xrange(nr):
		for i,noise in enumerate(noise_lev):
			pe[i,j],fe[i,j],nv[i,j] = compare_fa_pca(n,p,r,noise)

	# plot error as a function of noise
	plt.figure()
	plt.plot(nv,pe,'.b',label='pca')
	#plt.plot(noise_lev,se,'.r',label='standardized pca')
	plt.plot(nv,fe,'.r',label='factor analysis')
	plt.xscale('log')
	plt.xlabel('variance in noise levels across features')
	plt.savefig('pca_fa_comparison.eps')
	
	plt.show()

def compare_fa_pca(n,p,r,nv):
	# generate data from factor analysis model
	data,components,noise_variance = gen_data(n,p,r,noise_var=nv)

	# perform pca (on z-scored data) and factor analysis
	pca_c = do_pca(data,r)
	fa_c,noise_est = do_factor_analysis(data,r)

	# error is the mean cannonical correlation angle between subspaces
	pca_error = 1 - np.mean(canoncorr(components,pca_c))
	fa_error = 1 - np.mean(canoncorr(components,fa_c))

	return pca_error,fa_error,np.std(noise_variance)

def do_pca(A,r,standardize=True):
	"""
    Performs principal components analysis.
    
    Parameters
    ----------
    A : array_like
    	n x p matrix holding n observations of p features
    standardize : bool
    	if True, z-scores each column of A
    r : int
    	number of principal components to keep (or "rank" of reconstruction)

    Returns
    -------
    C : array_like
        r x p matrix holding the principal components in each row
    """

	if standardize:
	 	U,s,Vt = np.linalg.svd(A) #/ np.std(A,axis=0))
	else:
	 	U,s,Vt = np.linalg.svd(A)

	# principal components are just right singular vectors, Vt
	return Vt[:r,:].T

def do_factor_analysis(A,r=None):
	if r is None: r = gavish_threshold(A)
	fa = FactorAnalysis(n_components=r)
	fa.fit(A)

	return fa.components_.T, fa.noise_variance_

def gen_data(n,p,r,noise_var=1.0,noise_scale=1.0):
	"""
    Generates data from factor analysis.
    
    Parameters
    ----------
    n,p,r : int
        number of observations (n), features (p), and components (r)
    noise_scale : float
    	scales the noise for all features
    
    Returns
    -------
    A : array_like
        n x p matrix holding n observations of p features
    C : array_like
    	r x p matrix holding r ground truth components/factors
    psc : array_like
    	vector holding ground truth variance for each of p features
    """

	# low rank subspace
	W = np.random.randn(n,r)
	C = np.random.randn(p,r)
	A = np.dot(W,C.T)
	
	# add noise
	psc = noise_scale * np.exp(np.random.randn(1,p) * np.sqrt(noise_var))
	A += np.sqrt(psc) * np.random.randn(n,p)
	return A,C,psc

def canoncorr(X, Y):
    """
    Canonical correlations between two subspaces (computed via the QR
    decomposition). Function by Niru Maheswaranathan (@nirum).
    
    Parameters
    ----------
    X, Y : array_like
        The subspaces to compare. They should be of the same size.
    Returns
    -------
    corr : array_like
        The magnitude of the overlap between each dimension of the subspace.
    """

    # Orthogonalize each subspace
    qu, qv = np.linalg.qr(X)[0], np.linalg.qr(Y)[0]

    # singular values of the inner product between the orthogonalized spaces
    return np.linalg.svd(qu.T.dot(qv), compute_uv=False, full_matrices=False)


if __name__ == '__main__':
	main()
