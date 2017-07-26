import random

import numpy as np
from matplotlib import pyplot as plt


def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = random.randint(1, 9) + 0.5
        X[ii][1] = random.randint(1, 9) + 0.5
        Y[ii] = 1 if X[ii][0] + X[ii][1] >= 13 else -1
    return X, Y


X, Y = generate_data(100)
pos, neg = X[Y == 1, :], X[Y == -1, :]

plt.scatter(pos[:, 0], pos[:, 1], marker='o', c='blue')
plt.scatter(neg[:, 0], neg[:, 1], marker='x', c='red')

plt.plot(np.arange(10), np.arange(10))
plt.show()
