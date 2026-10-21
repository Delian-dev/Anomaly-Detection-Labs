import numpy as np
import matplotlib.pyplot as plt
from pyod.models.knn import KNN

from sklearn.metrics import confusion_matrix
from pyod.utils.data import generate_data_clusters

def calcul_BA(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2

def plotare(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], c='blue', label='Inliers', alpha=0.5, s=15)
    ax.scatter(X[y == 1, 0], X[y == 1, 1], c='red', label='Outliers', alpha=0.5, s=15)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.legend()
    ax.set_title(title)

contamination = 0.1

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200,
                                                            n_features=2, contamination=contamination, 
                                                            n_clusters=2, random_state=42)


for k in [1,3,5,10,25,50]:
    clf = KNN(n_neighbors=k, contamination=contamination)
    clf.fit(X_train)
    y_pred_train, y_pred_test = clf.predict(X_train), clf.predict(X_test)

    ba_train = calcul_BA(y_train, y_pred_train)
    ba_test = calcul_BA(y_test, y_pred_test)
    print(f"KNN={k}: BA train={ba_train:.3f}, BA test={ba_test:.3f}")

    fig, axe = plt.subplots(2,2, figsize=(10,8))
    plotare(axe[0, 0], X_train, y_train, 'Train data')
    plotare(axe[0, 1], X_train, y_pred_train, 'Train - predicted')
    plotare(axe[1, 0], X_test, y_test, 'Test data')
    plotare(axe[1, 1], X_test, y_pred_test, 'Test - predicted')
    axe[0, 0].legend()

    fig.suptitle(f'KNN, n_neighbors={k}')

    plt.tight_layout() #fara asta se vede urat

plt.show()