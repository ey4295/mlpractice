import random

import numpy as np
from matplotlib import pyplot as plt


# def generate_data(no_points):
#     X = np.zeros(shape=(no_points, 2))
#     Y = np.zeros(shape=no_points)
#     for ii in range(no_points):
#         X[ii][0] = random.randint(1, 9) + 0.5
#         X[ii][1] = random.randint(1, 9) + 0.5
#         Y[ii] = 1 if X[ii][0] + X[ii][1] >= 13 else -1
#     return X, Y
#
#
# X, Y = generate_data(100)
# pos, neg = X[Y == 1, :], X[Y == -1, :]
#
# plt.scatter(pos[:, 0], pos[:, 1], marker='o', c='blue')
# plt.scatter(neg[:, 0], neg[:, 1], marker='x', c='red')
#
# plt.plot(np.arange(10), np.arange(10))
# plt.show()

########################################################################################################
# matplot tutorial
#
########################################################################################################
"""
=====
Decay
=====

This example showcases a sinusoidal decay animation.
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def data_gen(t=0):
    cnt = 0
    while cnt < 1000:
        cnt += 1
        t += 0.1
        yield t, np.sin(2*np.pi*t) * np.exp(-t/10.)


def init():
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlim(0, 10)
    del xdata[:]
    del ydata[:]
    line.set_data(xdata, ydata)
    return line,

fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)#tuple with 1 value
ax.grid()
xdata, ydata = [], []


def run(data):
    # update the data
    t, y = data
    xdata.append(t)
    ydata.append(y)
    xmin, xmax = ax.get_xlim()

    if t >= xmax:
        ax.set_xlim(xmin, 2*xmax)
        ax.figure.canvas.draw()
    line.set_data(xdata, ydata)

    return line,

ani = animation.FuncAnimation(fig, run, data_gen, blit=False, interval=10,
                              repeat=False, init_func=init)
plt.show()