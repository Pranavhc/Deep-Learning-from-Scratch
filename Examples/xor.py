import numpy as np
import matplotlib.pyplot as plt

from nn.network import NeuralNetwork
from nn.optim import Adam
from nn.losses import BinaryCrossEntropy as BCE
from nn.layers import Dense
from nn.activations import Sigmoid
from nn.utils import DataLoader

X = np.reshape([[0,0], [0, 1], [1, 0], [1, 1]], (4, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1))

data = DataLoader(X, Y, 4, True)

model = NeuralNetwork(Adam(), BCE(), [
    Dense(2, 4), 
    Sigmoid(),
    Dense(4, 2), 
    Sigmoid(),
    Dense(2, 1), 
    Sigmoid()
])

model.fit(data, epochs=800, verbose=False)

###### Decision Boundary
def decision_boundary():
    points = []
    for x in np.linspace(0, 1, 20):
        for y in np.linspace(0, 1, 20): z = model.predict(np.array([[x, y]]))[0, 0]; points.append([x, y, z])
    points = np.array(points)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    plt.show()
decision_boundary()