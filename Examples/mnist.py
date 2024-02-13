import numpy as np
from keras.datasets import mnist
import matplotlib.pyplot as plt

import sys; sys.path.append('D:/Python/Deep Learning/Neural-Network') # ignore this

from dense import Dense
from activation import Sigmoid, Softmax, ReLu
from loss_fn import CrossEntropy
from network import NeuralNetwork
from regularization import Regularization


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
x_test, y_test = preprocess_data(x_test, y_test, 1000) 

# neural network
network = NeuralNetwork([
    Dense(28 * 28, 80, regularization=Regularization('l2', 0.0001)), 
    Sigmoid(),  
    Dense(80, 40, regularization=Regularization('l2', 0.0001)),  
    Sigmoid(),
    Dense(40, 25, regularization=Regularization('l2', 0.00001)),
    Sigmoid(),
    Dense(25, 10, regularization=None),
    Softmax()
])

network.train(CrossEntropy(), x_train, y_train, epochs=180, learning_rate=0.1, verbose_interval=10) # experiment with these couple of hyperparameters

def calc_accuracy(x_test, y_test):
    correct_count = 0
    for x, y in zip(x_test, y_test):
        if np.argmax(network.predict(x)) == np.argmax(y): 
            correct_count += 1
    return (correct_count / len(y_test)) * 100
    
print(f"\nAccuracy: {calc_accuracy(x_test, y_test)}%" ) 

def plot_images_with_predictions():
    fig = plt.figure(figsize=(8, 5))  # figure size in inches
    
    for i in range(50): 
        plt.subplot(5, 10, i+1)  # arrange the plots in a 5x10 grid
        plt.xticks([]); plt.yticks([])  # remove markings
        
        img = x_test[i].reshape((28, 28))
        plt.imshow(img, cmap=plt.cm.binary)
        
        predicted_label = np.argmax(network.predict(x_test[i]))
        true_label = np.argmax(y_test[i])
        
        color = 'green' if predicted_label == true_label else 'red'
        
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)  # set the label below the image
    plt.show()

plot_images_with_predictions()
# --------------------------------------------------------------------------------
# accuracy = ~91% (hyperparameters being e=300, a=0.2, samples=10000, layers=8, no regularization)
# best accuracy with current hyperparameters: ~90%

