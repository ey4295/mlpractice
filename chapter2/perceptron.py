"""
implementation of Perceptron Learning Algorithm(PLA)
"""
from random import sample, randint

import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split


def get_misclassifications(X_train, y_train, w, b):
    """
    find all misclassifications given w,b
    :param X_train:
    :param y_train:
    :param w:
    :param b:
    :return:
    """
    return [(x, y_train[i]) for (i, x) in enumerate(X_train) if y_train[i] * (x.dot(np.transpose(w)) + b) <= 0]


def PLA_decision(x, w, b):
    """
    decision function of model PLA
    :param X:array of input features
    :param w:trained parameter
    :param w:trained parameter
    :return:
    """
    return np.sign(x.dot(np.transpose(w)) + b)


def PLA_eval(X_test, y_test, w, b):
    """
    evaluation
    :param X_test:
    :param y_test:
    :return:
    """
    predictions = np.array([1 if PLA_decision(x, w, b) == y_test[i] else 0 for (i, x) in enumerate(X_test)])
    print('preditions are {}'.format(predictions))
    print('true values are {}'.format(y_test))

    return sum(predictions) / float(len(predictions))


def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = randint(1, 9) + 0.5
        X[ii][1] = randint(1, 9) + 0.5
        Y[ii] = 1 if X[ii][0] + X[ii][1] >= 13 else -1
    return X, Y


def visualization(X, y, params):
    """
    plot figures and enact animation
    :param X: arrray of input features
    :param y: array of labels
    :param params: traned params
    :return:
    """
    fig, ax = plt.subplots()
    pos, neg = X[y == 1, :], X[y == -1, :]
    print(pos, neg)
    line, = ax.plot(np.zeros(np.shape(X)[1]), np.zeros(np.shape(X)[1]))

    def init():
        """
        init for animation
        :return:
        """
        pos_points = ax.scatter(pos[:, 0], pos[:, 1])
        neg_points = ax.scatter(neg[:, 0], neg[:, 1])

    def update(data):
        """
        update each frame
        :return:
        """
        w, b = data
        line.set_data(np.linspace(0, 10, num=100), -1 * (w[0] * np.linspace(0, 10, num=100) + b) / w[1])
        # ax.figure.canvas.draw()
        return line

    ani = animation.FuncAnimation(fig, update, params, init_func=init, interval=500, repeat=False)
    plt.show()


def PLA_main():
    """
    main function of PLA
    :return:
    """
    # read data
    # digits = datasets.load_digits(n_class=2)
    # X, y = digits.get('data'), digits.get('target')

    X, y = generate_data(100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    pos, neg = X[y == 1, :], X[y == -1, :]

    # train
    ## 1 initialize
    w, b, ita = np.zeros(np.shape(X_train)[1], dtype=float), 0.0, 1.0
    bad_x, bad_y = sample(get_misclassifications(X_train, y_train, w, b), 1)[0]
    ## 2 update according to Gradient Descent Methods
    plotdata = []
    while not (all(np.isclose(w, w + ita * bad_y * bad_x)) and np.isclose(b, b + ita * bad_y)):
        w = w + ita * bad_y * bad_x
        b = b + ita * bad_y
        plotdata.append((w, b))
        if not get_misclassifications(X_train, y_train, w, b):
            break
        else:
            print('{} bad points'.format(len(get_misclassifications(X_train, y_train, w, b))))
            bad_x, bad_y = sample(get_misclassifications(X_train, y_train, w, b), 1)[0]

        print('w = {}'.format(w))
        print('b = {}'.format(b))

    print('############ final w = {}'.format(w))
    print('############ final b= {}'.format(b))
    visualization(X, y, plotdata)

    # prediction
    score = PLA_eval(X_test, y_test, w, b)
    print('score = {}'.format(score))


PLA_main()
