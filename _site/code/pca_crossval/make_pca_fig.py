from pca_crossval import * 
import itertools
from tqdm import tqdm
np.random.seed(2222)

m, n, r = 500, 501, 5
Utrue, Vtrue = randn(m,r), randn(r,n)
data = np.dot(Utrue, Vtrue) + 6*randn(m,n)
ranks = []
train_err = []
test_err = []

rank_range = range(1,11)
repeat_range = range(10)
with tqdm(total=len(rank_range)*len(repeat_range)) as pbar:
    for rank, repeat in itertools.product(rank_range, repeat_range):
        _, _, train, test = crossval_pca(data, rank)
        ranks.append(rank)
        train_err.append(train[-1])
        test_err.append(test[-1])
        pbar.update(1)
ranks = np.array(ranks)
train_err = np.array(train_err)
test_err = np.array(test_err)

fig, axes = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(7, 4))
axes[0].plot(ranks + randn(ranks.size)*.1, train_err, '.k')
axes[0].plot(np.unique(ranks), [np.mean(train_err[ranks==r]) for r in np.unique(ranks)], '-r', zorder=-1)
axes[0].set_ylabel('RMSE')
axes[0].set_title('Train Error')
axes[0].set_xlabel('# of components')

axes[1].plot(ranks + randn(ranks.size)*.1, test_err, '.k')
axes[1].plot(np.unique(ranks), [np.mean(test_err[ranks==r]) for r in np.unique(ranks)], '-r', zorder=-1)
axes[1].set_xticks(np.unique(ranks).astype(int))
axes[1].set_title('Test Error')
axes[1].set_xlabel('# of components')

fig.tight_layout()
fig.savefig('cv_pca.png', dpi=500)
plt.show()
