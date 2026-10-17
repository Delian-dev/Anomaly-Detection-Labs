from pyod.models.knn import KNN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve

from pyod.utils.data import generate_data


def calcul_formule(y_true, y_pred):
    #calcul matrice de confuzie
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

    tpr = tp/(tp + fn)  # true ppositive
    tnr = tn/(tn +fp)  # true negative
    balanced_accuracy = (tpr+tnr)/2 #ballanced accuracy
    
    print(f"TPR: {tpr:.2f}, TNR: {tnr:.2f}, Balanced Accuracy: {balanced_accuracy:.2f}")

contamination = 0.1  # de la ex1

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100,
                                                n_features=2, contamination=contamination, random_state=42)

classifier = KNN(contamination=contamination)
classifier.fit(X_train)

#0 pt inlinier, 1 pt oulier
y_train_pred = classifier.predict(X_train)
y_test_pred = classifier.predict(X_test)


calcul_formule(y_train, y_train_pred)
calcul_formule(y_test, y_test_pred)

test_scores = classifier.decision_function(X_test) #decision function in loc de predict pt ca roc curve are nevoie de scoruri de continue, nu de predictii 0/1
#calcul ROC curve pe baza scorurilor de decizie
fpr, tpr, _ = roc_curve(y_test, test_scores)


#mutata inainte ca sa nu se suprapuna cu plotarea
print("Teste cu alte grade de contaminare:")
for contamination in [0.01, 0.05, 0.1, 0.2, 0.3, 0.5]: #se observa cum pana intre 0.1 si 0.3 TPR ramane la constant, dar TNR scade => BA scade
    model = KNN(contamination=contamination)
    model.fit(X_train)
    y_test_pred = model.predict(X_test)
    print(f"Contamination: {contamination}")
    calcul_formule(y_test, y_test_pred)
    print("-" * 50)
#BA maxim se atinge pt contamination = 0.1

#plotare
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label="KNN ROC curve")
plt.plot([0, 1], [0, 1], 'k--', label="random")

plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.title("ROC Curve - KNN Anomaly Detection")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.show()
