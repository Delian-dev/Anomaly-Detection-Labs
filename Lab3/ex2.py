import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

contamination = 0.02

def genereaza(centers, n_features):
    # 2 clustere, 500 de puncte fiecare, std 1
    X_train, _ = make_blobs(n_samples=[500, 500], n_features=n_features,
                      centers=centers, cluster_std=1, random_state=42)
    X_test = np.random.uniform(-10, 20, (1000, n_features))
    return X_train, X_test


def harta(ax, model, nume, X_train, X_test, dim):
    model.fit(X_train)
    scores = model.decision_function(X_test)   # mai mare = mai anormal
    if dim == 2:
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap='viridis', s=15)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
    else:
        sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores,
                        cmap='viridis', s=15)
        ax.set_xlabel('x1')
        ax.set_ylabel('x2')
        ax.set_zlabel('x3')

    ax.set_title(nume)
    plt.colorbar(sc, ax=ax)

def calcul_date(centers, dim, hidden=None, n_bins=10):
    X_train, X_test = genereaza(centers, dim)
    modele = [
        ('IForest', IForest(contamination=contamination)),
        ('DIF', DIF(contamination=contamination, hidden_neurons=hidden)),
        ('LODA', LODA(contamination=contamination, n_bins=n_bins)),
    ]
    
    fig = plt.figure(figsize=(15, 5))
    for i, (nume, model) in enumerate(modele):
        if dim == 2:
            ax = fig.add_subplot(1, 3, i + 1)
        else: #cazul 3D
            ax = fig.add_subplot(1, 3, i + 1, projection='3d')
        harta(ax, model, nume, X_train, X_test, dim)

    fig.suptitle(f'{dim}D - DIF hidden={hidden}, LODA n_bins={n_bins}')
    plt.tight_layout()


import warnings
warnings.filterwarnings("ignore", category=UserWarning) #pt ca primesc multe de la torch cand ruleaza pe cpu

#Cazul 2D
c2 = [(10, 0), (0, 10)]
calcul_date(c2, 2)                        
calcul_date(c2, 2, hidden=[16, 8], n_bins=5) 
calcul_date(c2, 2, hidden=[128, 64], n_bins=50)

#3D
c3 = [(0, 10, 0), (10, 0, 10)]
calcul_date(c3, 3)  
calcul_date(c3, 3, hidden=[16, 8], n_bins=5)
calcul_date(c3, 3, hidden=[128, 64], n_bins=50)

plt.show()