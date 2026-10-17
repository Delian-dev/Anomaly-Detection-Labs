import numpy as np
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix

def calcul_formule(y_true, y_pred):
    #calcul matrice de confuzie
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    tpr = tp/(tp + fn)  # true ppositive
    tnr = tn/(tn +fp)  # true negative
    balanced_accuracy = (tpr+tnr)/2 #ballanced accuracy
    
    return tn, fp, fn, tp, tpr, tnr, balanced_accuracy

contamination = 0.1

rng = np.random.default_rng(42)
n_samples, dim = 1000, 3

mu = np.array([1.0, -2.0, 3.0])
Sigma = np.array([[2.0, 0.8, 0.3],
                  [0.8, 1.5, 0.5],
                  [0.3, 0.5, 1.0]])

L = np.linalg.cholesky(Sigma)

n_outliers = int(contamination * n_samples)
n_inliers = n_samples - n_outliers

#inliers - y = Lx + mu
x_inliers = rng.standard_normal((n_inliers, dim))
y_inliers = x_inliers @ L.T + mu 

#outlies - am incercat cu mai multe constante
x_outliers = 3.0 *rng.standard_normal((n_outliers, dim))
#x_outliers = 5.0 * rng.standard_normal((n_outliers, dim))
#x_outliers = rng.standard_normal((n_outliers, dim)) #barely peste 50%
y_outliers = x_outliers @ L.T + mu

X = np.vstack((y_inliers, y_outliers))
y_true = np.concatenate([np.zeros(n_inliers), np.ones(n_outliers)])

# Z-score multidimensional: z = L^-1 (X-mu)
mu_hat = X.mean(axis=0)
Sigma_hat = np.cov(X, rowvar=False) 
L_hat = np.linalg.cholesky(Sigma_hat)
Z_score = np.linalg.solve(L_hat, (X - mu_hat).T).T

scores = np.linalg.norm(Z_score, axis=1) #norma euclidiana vector z-scort pt fiecare punct

threshold = np.quantile(scores, 1 - contamination)
y_pred = (scores > threshold).astype(int)

tn,fp,fn,tp,tpr,tnr,ba = calcul_formule(y_true, y_pred)

print(f"Threshold z-score: {threshold:.2f}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"TPR: {tpr:.2f}, TNR: {tnr:.2f}, Balanced Accuracy: {ba:.2f}")