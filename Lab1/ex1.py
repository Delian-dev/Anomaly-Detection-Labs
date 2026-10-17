import matplotlib.pyplot as plt
from pyod.utils.data import generate_data

X_train, X_test, y_train, y_test = generate_data(n_train=400, n_test=100, 
                                                n_features=2, contamination=0.1, random_state=42)

#print(y_train, y_test)

train_inliers = X_train[y_train==0]
train_outliers = X_train[y_train==1]

#test plotare
plt.figure(figsize=(8, 6))

plt.scatter(
    train_inliers[:, 0],
    train_inliers[:, 1],
    c="blue",
    alpha=0.5,
    label="Inliers"
)

plt.scatter(
    train_outliers[:, 0],
    train_outliers[:, 1],
    c="red",
    edgecolors="k",
    alpha=0.5,
    label="Outliers"
)

plt.title("Training Data Distribution")
plt.xlabel("Feature 1 - X1")
plt.ylabel("Feature 2 - X2")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.show()