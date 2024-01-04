import numpy as np
from keras.datasets import mnist
from keras.src.utils.np_utils import to_categorical

import sys
sys.path.append('D:/Python/Deep Learning/Neural-Network')

from dense import Dense
from activation import Sigmoid
from loss_fn import mse, mse_prime
from network import train, predict


def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    # encode output which is a number in range [0,9] into a vector of size 10
    # e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = to_categorical(y)
    y = y.reshape(y.shape[0], 10, 1)
    return x[:limit], y[:limit]


# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 1000)
x_test, y_test = preprocess_data(x_test, y_test, 20)

# neural network
network = [
    Dense(28 * 28, 40),
    Sigmoid(),
    Dense(40, 10),
    Sigmoid()
]

# train
train(network, mse, mse_prime, x_train, y_train, epochs=2000, learning_rate=0.1)

# test
for x, y in zip(x_test, y_test):
    output = predict(network, x)
    print('pred:', np.argmax(output), '\ttrue:', np.argmax(y), '\t', np.argmax(output)==np.argmax(y))