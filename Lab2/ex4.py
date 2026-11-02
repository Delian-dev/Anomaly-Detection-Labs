import numpy as np
from sklearn.metrics import confusion_matrix
from pyod.models.knn import KNN

from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

from sklearn.model_selection import train_test_split

def calcul_BA(y_true, y_pred):
    #calcul matrice de confuzie
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp/(tp + fn)  # true ppositive
    tnr = tn/(tn +fp)  # true negative
    balanced_accuracy = (tpr+tnr)/2 #ballanced accuracy
    
    return balanced_accuracy

from scipy.io import loadmat #pt citire fisiere matlab

# incarcare date
data = loadmat('Lab2/cardio.mat')
X = data['X']
y = data['y'].ravel()
contamination = y.sum() / len(y) 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42) #mai testez si alte test_size

# cele 10 modele KNN cu n_neighbors de la 30 la 120
ks = np.linspace(30, 120, 10).astype(int)
train_scores = np.zeros((len(X_train), len(ks)))
test_scores = np.zeros((len(X_test), len(ks)))

for i, k in enumerate(ks):
    clf = KNN(n_neighbors=k, contamination=contamination)
    clf.fit(X_train)
    train_scores[:, i] = clf.decision_scores_ #scorurile de anomalii pentru datele de antrenament
    test_scores[:, i] = clf.decision_function(X_test) #scorurile de anomalii pentru datele de test

    ba_train = calcul_BA(y_train, clf.predict(X_train))
    ba_test = calcul_BA(y_test, clf.predict(X_test))
    print(f"k={k}: BA train={ba_train:.3f}, BA test={ba_test:.3f}")


train_scores_norm, test_scores_norm = standardizer(train_scores, test_scores) #standardizare scoruri per model

#comvinarea scorurilor prin medie si maximizare
for nume,comb in [('average', average), ('maximization', maximization)]:
    tr = comb(train_scores_norm)
    te = comb(test_scores_norm)

    # prag cuantila
    prag = np.quantile(tr, 1 - contamination)
    y_train_pred = (tr > prag).astype(int)
    y_test_pred = (te > prag).astype(int)

    print(f"{nume}: BA for Train={calcul_BA(y_train, y_train_pred):.3f}, "
          f"BA for Test={calcul_BA(y_test, y_test_pred):.3f}")