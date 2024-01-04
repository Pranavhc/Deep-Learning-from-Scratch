import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
sys.path.append('D:/Python/Deep Learning/Neural-Network')

from dense import Dense
from activation import Sigmoid, ReLu
from loss_fn import mse, mse_prime
from network import train, predict

X = np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    ReLu(),
    Dense(3, 1),
    Sigmoid()
]

train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1, verbose_interval=100)

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()