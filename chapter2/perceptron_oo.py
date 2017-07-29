"""
Object oriented version of Perceptron Learning algorithm
"""
import random

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


class PLA:
    """implementation of Percetron Learning Algorithm"""
    history = []

    def __init__(self, init_w, init_b, learning_rate=1):
        self.learning_rate = learning_rate
        self.w = init_w
        self.b = init_b

    def decision(self, x, w, b):
        """
        decision function
        :param x: array of input features
        :param w: array of parameters
        :param b: bias-parameter
        :return: y
        """
        return np.sign(x.dot(np.transpose(w)) + b)

    def train(self, train_X, train_Y):
        """
        tran the model
        :return:
        """
        self.X = train_X
        self.Y = train_Y

        def get_badpoints(w, b):
            """
            find illly classified points given w,b
            :param w: array of parameters
            :param b: bias-parameter
            :return: [(bad_x,bad_y)]
            """
            return [(x, y) for (x, y) in zip(self.X, self.Y) if y * (x.dot(np.transpose(w)) + b) <= 0]

        # loop
        self.history = []
        while get_badpoints(self.w, self.b):
            bad_x, bad_y = random.sample(get_badpoints(self.w, self.b), 1)[0]
            self.w += self.learning_rate * bad_x * bad_y
            self.b += self.learning_rate * bad_y
            print('w={}'.format(self.w))
            print('b={}'.format(self.b))
            self.history.append((self.w, self.b))

    def predict(self, test_X):
        """
        predict label for x
        :param x: array of input feature
        :return:
        """
        return [np.sign(x.dot(np.transpose(self.w))) for x in test_X]

    def vis_datasets(self, X, Y):
        """
        visualizationo of datasets
        :return:
        """
        fig, ax = plt.subplots()
        pos, neg = X[Y == 1, :], X[Y == -1, :]
        ax.scatter(pos[:, 0], pos[:, 1])
        ax.scatter(neg[:, 0], neg[:, 1])
        return ((fig, ax))

    def vis_train_process(self, X, Y):
        """
        visualization of training process
        :return
        """
        fig, ax = plt.subplots()
        line, = ax.plot(np.zeros(np.shape(X)[1]), np.zeros(np.shape(X)[1]))

        def init():
            """
            init function for visualization
            :return:
            """
            pos, neg = X[Y == 1, :], X[Y == -1, :]
            ax.scatter(pos[:, 0], pos[:, 1])
            ax.scatter(neg[:, 0], neg[:, 1])

        def update(data):
            """
            update each time
            :param data: iterated data
            :return:
            """
            w, b = data
            line.set_data(np.linspace(0, 10, num=100), -1 * (w[0] * np.linspace(0, 10, num=100) + b) / w[1])
            return line

        # must has a variable
        ani = animation.FuncAnimation(fig, update, self.history, init_func=init, interval=500, repeat=False)
        plt.show()

    def eval(self, test_X, test_Y):
        """
        evaluation of this classification problem
        :return:
        """
        return sum(
            [1 if np.sign(self.w.dot(np.transpose(x)) + self.b) == y else 0 for x, y in zip(test_X, test_Y)]) / float(
            len(test_Y))


def data_gen(no_points):
    """
    generate linear separable data
    :return:
    """
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = random.randint(1, 9) + 0.5
        X[ii][1] = random.randint(1, 9) + 0.5
        Y[ii] = 1 if X[ii][0] + X[ii][1] >= 13 else -1
    return X, Y


X, Y = data_gen(100)
print(Y)
train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.1)
classifier = PLA(np.zeros(2, dtype=float), 0.0)

classifier.train(train_X, train_Y)
classifier.vis_train_process(X, Y)
score = classifier.eval(test_X, test_Y)
print('score ={}'.format(score))
