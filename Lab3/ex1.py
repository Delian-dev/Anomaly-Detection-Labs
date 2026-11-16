import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_blobs

def loda_scores(X_train, X_test, W, n_bins):
    scores = np.zeros(len(X_test))
    for w in W:
        proj_train = X_train @ w
        proj_test = X_test @ w

        # range mai larg decat valorile proiectate (si ca sa acopere si testul)
        lim = 2 * np.abs(np.concatenate([proj_train, proj_test])).max()
        hist, edges = np.histogram(proj_train, bins=n_bins, range=(-lim, lim))
        prob = hist / hist.sum()   # probabilitatea fiecarui bin

        # bin-ul in care cade fiecare punct de test
        idx = np.digitize(proj_test, edges) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        scores += prob[idx]
    return scores / len(W)   # media probabilitatilor


rng = np.random.default_rng(42)

#500 de puncte 2D - normal standard
X_train, _ = make_blobs(n_samples=500, n_features=2, centers=[(0,0)], 
                        cluster_std=1.0, random_state=42)

vect_proj = 5
W = np.random.multivariate_normal([0,0], np.eye(2), size=vect_proj)
W = W / np.linalg.norm(W, axis=1, keepdims=True)  # normalizare

#datele de test - puncte in [-3,3]
X_test = rng.uniform(low=-3, high=3, size=(500, 2))

#testare pe bin-uri diferite
bins_list = [5, 10, 20]
fig, axe = plt.subplots(1, 3, figsize=(15, 4))
for ax, n_bins in zip(axe, bins_list):
    s = loda_scores(X_train, X_test, W, n_bins)
    sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=s, cmap='viridis', s=15)
    ax.set_title(f'n_bins={n_bins}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    plt.colorbar(sc, ax=ax)
fig.suptitle('Harta de scoruri - LODA (simplificat)')

plt.tight_layout()
plt.show()