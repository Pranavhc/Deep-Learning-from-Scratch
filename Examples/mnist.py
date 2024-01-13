import numpy as np
from keras.datasets import mnist
from keras.src.utils.np_utils import to_categorical

import sys; sys.path.append('D:/Python/Deep Learning/Neural-Network') # ignore this

from dense import Dense
from activation import Sigmoid
from loss_fn import mse, mse_prime
from network import train, predict
from regularization import Regularization

def preprocess_data(x, y, limit):
    # reshape and normalize input data
    x = x.reshape(x.shape[0], 28 * 28, 1)
    x = x.astype("float32") / 255
    
    y = to_categorical(y) # one-hot encoding - number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    y = y.reshape(y.shape[0], 10, 1)
    
    return x[:limit], y[:limit]

# load MNIST from keras datasets server
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, y_train = preprocess_data(x_train, y_train, 2000) # decrement sample count for quicker training
x_test, y_test = preprocess_data(x_test, y_test, 100)

# neural network
network = [
    Dense(28 * 28, 80, regularization=Regularization("L2", 0.01)),
    Sigmoid(),
    Dense(80, 40, regularization=Regularization("L2", 0.001)),
    Sigmoid(),
    Dense(40, 25, regularization=Regularization("L2", 0.001)),
    Sigmoid(),
    Dense(25, 10, regularization=Regularization("L2", 0.001)),
    Sigmoid()
]

# this will take a long time, reduce smaple count for quicker training
train(network, mse, mse_prime, x_train, y_train, epochs=300, learning_rate=0.2, verbose_interval=10) # experiment with these couple of hyperparameters

def calc_accuracy(x_test, y_test):
    correct_count = 0

    for x, y in zip(x_test, y_test):
        if np.argmax(predict(network, x)) == np.argmax(y): 
            correct_count += 1
    return (correct_count / len(y_test)) * 100
    
print(f"Accuracy: {calc_accuracy(x_test, y_test)}%" ) 

# best I could get was 91% accuracy w/ minimal hyperparameter tuning. (epochs=300, alpha=0.2, training_samples=10000 , layers=8, no regularization)
