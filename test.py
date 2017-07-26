import random

import numpy as np
# a=np.ndarray([1,23,4])
# b=np.ndarray([1,23,4])
# print(a.dot(np.transpose(b)))
# print(np.shape(np.zeros(4)))
#
# a=np.array([[1,2],[3,4]])
# print(a)
# print(type(a))
# for (i,row) in enumerate(a):
#     print(row)

def generate_data(no_points):
    X = np.zeros(shape=(no_points, 2))
    Y = np.zeros(shape=no_points)
    for ii in range(no_points):
        X[ii][0] = random.randint(1,9)+0.5
        X[ii][1] = random.randint(1,9)+0.5
        Y[ii] = 1 if X[ii][0]+X[ii][1] >= 13 else -1
    return X, Y

print(generate_data(10))