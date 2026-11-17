import numpy as np
import matplotlib.pyplot as plt


from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from pyod.utils.utility import standardizer

def calcul_BA(y_true, y_pred): #functia de o tot plimb peste tot
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    return (tpr + tnr) / 2


# incarcare date
data = loadmat('Lab3/shuttle.mat')
X = data['X']
y = data['y'].ravel()
contamination = y.sum() / len(y) 

n_splits = 10
rezultate = {'IForest': [], 'LODA': [], 'DIF': []} #liste de (BA, AUC)

for i in range(n_splits):
    # 40% test, split diferit la fiecare iteratie
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=i)

    # normalizare: statistici din train, aplicate si pe test
    X_train, X_test = standardizer(X_train, X_test)

    modele = {
        'IForest': IForest(contamination=contamination),
        'LODA': LODA(contamination=contamination),
        'DIF': DIF(contamination=contamination),
    }

    for nume, model in modele.items():
        model.fit(X_train)
        y_pred = model.predict(X_test)  #etichete 0/1
        scores = model.decision_function(X_test)  #scoruri continue

        ba = calcul_BA(y_test, y_pred)
        auc = roc_auc_score(y_test, scores)
        rezultate[nume].append((ba, auc))
        print(f"split {i + 1}: {nume} BA={ba:.2f}, AUC={auc:.2f}")

    print(f"split {i + 1}/{n_splits} done")
    print("-" * 40)

print("Medii pt 10 split-uri:")
for nume, vals in rezultate.items():
    vals = np.array(vals)
    print(f"{nume}: BA={vals[:, 0].mean():.2f}, ROC AUC={vals[:, 1].mean():.2f}")