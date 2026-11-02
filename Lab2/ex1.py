import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n_reg, n_sp, n_top = 100, 4, 5
nume = ['regular', 'varianta mare - X', 'varianta mare - Y', 'varianta mare - X si Y']
culori = ['blue', 'green', 'orange', 'purple']
zgomote = [(0, 0.1), (0, 1), (1, 3)]  # (mu, sigma^2)


def leverage(X):
    Xb = np.hstack([X, np.ones((len(X), 1))]) #adaugcoloana de 1 pentru bias
    H = Xb @ np.linalg.inv(Xb.T @ Xb) @ Xb.T # y prezis de regresie
    return np.diag(H)


def genereaza(w, b, mu, sigma2):
    d = len(w)
    s = np.sqrt(sigma2)

    #pozitiile pentru fiecare tip de punct
    X = [rng.normal(0, 1, (n_reg, d)),   # 0: regulate (100 pct): x normal
     rng.normal(0, 5, (n_sp, d)),    # 1: varianta mare pe x (4 pct): x risipit
     rng.normal(0, 1, (n_sp, d)),    # 2: varianta mare pe y (4 pct): x normal
     rng.normal(0, 5, (n_sp, d))]    # 3: varianta mare pe x si y (4 pct): x risipit

    # zgomotul pentru fiecare tip de punct
    e = [rng.normal(mu, 0.1, n_reg),     # 0: zgomot mic fix
     rng.normal(mu, s, n_sp),        # 1: zgomot normal (std = sqrt(sigma2))
     rng.normal(mu, 5 * s, n_sp),    # 2: zgomot mare (de 5 ori)
     rng.normal(mu, 5 * s, n_sp)]    # 3: zgomot mare (de 5 ori)

    
    tip = np.concatenate([np.full(n_reg, 0), np.full(n_sp, 1),
                          np.full(n_sp, 2), np.full(n_sp, 3)])
    X = np.vstack(X)
    y = X @ w + b + np.concatenate(e)
    return X, y, tip


''' Cazul unidimensional => y=a*x1+b+e'''
fig, axe = plt.subplots(1, 3, figsize=(12, 6))
for ax, (mu, sigma2) in zip(axe, zgomote):
    X, y, tip = genereaza(np.array([2.0]), 1.0, mu, sigma2)
    top = np.argsort(leverage(X))[-n_top:]
    for i in range(4):
        ax.scatter(X[tip == i, 0], y[tip == i], c=culori[i], label=nume[i], alpha=0.5, s=15)

    ax.scatter(X[top, 0], y[top], s=120, facecolors='none', edgecolors='red', label='leverage maxim')
    ax.set_title(f'mu={mu}, sigma^2={sigma2}')
    ax.set_xlabel('x1')
    ax.set_ylabel('y')

fig.suptitle('Leverage scores - 1D')
plt.tight_layout()


''' Cazul bidimensional => y = a*x1 + b*x2 + c + e'''
fig, axe = plt.subplots(1, 3, figsize=(12, 6))
for ax, (mu, sigma2) in zip(axe, zgomote):
    X, y, tip = genereaza(np.array([2.0, -1.0]), 1.0, mu, sigma2)
    top = np.argsort(leverage(X))[-n_top:]
    for i in range(4):
        ax.scatter(X[tip == i, 0], X[tip == i, 1], c=culori[i], label=nume[i], alpha=0.5, s=15)

    ax.scatter(X[top, 0], X[top, 1], s=120, facecolors='none', edgecolors='red', label='leverage maxim')
    ax.set_title(f'mu={mu}, sigma^2={sigma2}')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')

fig.suptitle('Leverage scores - 2D')
plt.tight_layout()

plt.show()
