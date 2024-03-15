import numpy as np
import matplotlib.pyplot as plt

from nn.network import NeuralNetwork
from nn.dense import Dense
from nn.activation import Sigmoid, ReLu
from nn.loss_fn import MSE


X = np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = NeuralNetwork([
    Dense(2, 6, regularization=None), 
    ReLu(),  
    Dense(6, 3, regularization=None),
    ReLu(),
    Dense(3, 1, regularization=None),
    Sigmoid()
])

# network.train(MSE(), X, Y, epochs=10000, learning_rate=0.1, verbose_interval=100)
# network.save("Examples/models/xor.pkl")

network = NeuralNetwork([]).load("Examples/models/xor.pkl")

def decision_boundary():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20):
            z = network.predict([[x], [y]])
            points.append([x, y, z[0,0]])

    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
    plt.show()

decision_boundary()