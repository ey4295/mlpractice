"""
    90 exercises for numpy
"""

import numpy as np

print(np.__version__)
np.show_config()
Z = np.zeros(10)
print(Z)
Z[4] = 1
print(Z)
Z = np.arange(10, 49)
print(Z)

# reverse/list slice
print(Z[::-1])
print(Z[::3])

Z = np.arange(0, 9).reshape(3, 3)
print(Z)
nz = np.nonzero(Z)
print(nz)

print(np.nonzero([1, 20, 2, 0]))

#random matrix
print(np.eye(3))
z = np.random.random((3, 3, 3))
print(z)
z = np.random.random((3, 3))
print(z)
z = np.random.random(3)
print(z)

z=np.ones()

