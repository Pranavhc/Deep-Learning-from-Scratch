import numpy as np
from keras.datasets import mnist

import sys; sys.path.append('D:/Python/Deep Learning/Neural-Network') # ignore this

from dense import Dense
from activation import Sigmoid
from loss_fn import mse, mse_prime
from network import train, predict

def one_hot_encode(y: np.ndarray) -> np.ndarray:
    encoded = np.zeros((y.shape[0], 10))    
    encoded[np.arange(y.size), y] = 1 # encoded[<0 to len(y)>, <current value of y>] = 1
    return encoded

# an obscure way to do the same as above
def one_hot_encode1(y: np.ndarray) -> np.ndarray:
    return np.eye(10)[y]

# reshape and normalize input data
def preprocess_data(x, y, limit):
    x = x[:, :limit].reshape(x.shape[0], 28 * 28, 1) # flattening 28x28 matrix to get a 784 vector
    x = x.astype("float32") / 255 # now the data ranges between 0 and 1 (e.g. 0.5, 0.123, 0.882)
    
    # one-hot encoding (e.g. 3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
    y = one_hot_encode1(y[:limit])  
    y = y.reshape(y.shape[0], 10, 1) 
    
    return x, y

# load MNIST from keras datasets server
(x_train, y_train), (x_test, y_test) = mnist.load_data() 

x_train, y_train = preprocess_data(x_train, y_train, 4000) # using only 4k/60k samples for faster training
x_test, y_test = preprocess_data(x_test, y_test, 100) 

# neural network
network = [
    Dense(28 * 28, 80, regularization=None), 
    Sigmoid(),  
    Dense(80, 40),  
    Sigmoid(),
    Dense(40, 25),
    Sigmoid(),
    Dense(25, 10),
    Sigmoid()
]

train(network, mse, mse_prime, x_train, y_train, epochs=150, learning_rate=0.2, verbose_interval=10) # experiment with these couple of hyperparameters

def calc_accuracy(x_test, y_test):
    correct_count = 0
    for x, y in zip(x_test, y_test):
        if np.argmax(predict(network, x)) == np.argmax(y): 
            correct_count += 1
    return (correct_count / len(y_test)) * 100
    
print(f"Accuracy: {calc_accuracy(x_test, y_test)}%" ) 


# --------------------------------------------------------------------------------
# accuracy = ~91% (hyperparameters being e=300, a=0.2, samples=10000, layers=8, no regularization)
# best accuracy with current hyperparameters: ~88%