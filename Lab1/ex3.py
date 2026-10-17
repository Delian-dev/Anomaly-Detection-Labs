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

#generare doar esantioane de antrenament (1D)
X_train,_, y_train,_ = generate_data(n_train=1000, n_test=0, 
                                     n_features=1, contamination=contamination, random_state=42)

#calcul z-score
z = np.abs((X_train - X_train.mean()) / X_train.std()).ravel() #ravel pt a face din (1000,1) in (1000,)

#pragul de 10% marcat ca anomalie
threshold = np.quantile(z, 1 - contamination)
y_pred = (z > threshold).astype(int)


tn, fp, fn, tp, tpr, tnr, ba = calcul_formule(y_train, y_pred)

print(f"Threshold z-score: {threshold:.2f}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"TPR: {tpr:.2f}, TNR: {tnr:.2f}, Balanced Accuracy: {ba:.2f}")