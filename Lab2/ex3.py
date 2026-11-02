import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN
from pyod.models.lof import LOF

contamination = 0.07

from sklearn.datasets import make_blobs

X1, _ = make_blobs(n_samples=200, n_features=2, centers=[(-10, -10)],
                   cluster_std=2, random_state=42) #primul cluster cu 200 de samples; centru (-10, -10) si deviatia standard 2

X2, _ = make_blobs(n_samples=100, n_features=2, centers=[(10, 10)],
                   cluster_std=6, random_state=42) #al doilea cluster cu 100 de samples; centru = (10, 10), deviatia=6

X = np.vstack([X1, X2])

#comparam pt valori diferite ale lui k

for k in [1, 3, 5, 10, 25, 50]:
    knn = KNN(n_neighbors=k, contamination=contamination)
    knn.fit(X)
    
    lof = LOF(n_neighbors=k, contamination=contamination)
    lof.fit(X)

    fig, axe = plt.subplots(1, 2, figsize=(12, 5))
    for ax, model, name in zip(axe, [knn, lof], ['KNN', 'LOF']):
        y_pred = model.labels_ #etichetele prezise de model
        ax.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], c='blue', label='Inliers', alpha=0.5, s=15)
        ax.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], c='red', label='Outliers', alpha=0.5, s=15)
        ax.set_title(f'{name} - n_neighbors={k}')

    axe[0].legend()
    fig.suptitle(f'Comparatie modele KNN si LOF pt n_neighbors={k}')
    plt.tight_layout()

plt.show()