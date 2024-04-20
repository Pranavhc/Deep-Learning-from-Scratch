#### This doesn't work yet

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from nn.network import NeuralNetwork
from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Conv2D, Dense, Dropout, Flatten, Reshape
from nn.activations import ReLu, Softmax
from nn.utils import DataLoader, to_categorical1D, shuffler, save_object, load_object

################# PREPARE DATA 

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], 1, 28, 28) 
    x = x.astype("float32") / 255     # normalization: data ranges between 0 and 1
    y = to_categorical1D(y, 10)       # one-hot encoding
    return shuffler(x, y)             # return shuffled data

(train_data), (test_data) = mnist.load_data() 

X_train, y_train = preprocess_data(train_data[0], train_data[1])
X_test, y_test = preprocess_data(test_data[0], test_data[1]) 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

################# SET HYPERPARAMETERS

n_samples, channels, height, width = X_train.shape
learning_rate= 0.01
epochs=2
momentum= 0.9
batch_size= 12

################# DEFINE THE MODEL

clf = NeuralNetwork(Adam(), CCE(), [
    Conv2D(n_filters=3, filter_shape=(3, 3), input_shape=(channels, height, width)),
    ReLu(),
    
    # Reshape((3 * height * width)),
    Flatten(),

    Dense(3*28*28, 32), # this should receive a (128, 784) 2d array 
    Dropout(0.2),
    ReLu(),
    
    Dense(32, 10),
    Softmax()
])

################# DEFINE DATA LOADERS AND TRAIN THE MODEL

train_loader = DataLoader(X_train, y_train, batch_size, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size, shuffle=True)

history = clf.fit(train_loader, val_loader, epochs, verbose=True)


################# SAVE THE MODEL
save_object(clf, "Examples/models/mnist_conv_clf.pkl")

################# LOAD THE MODEL
clf = load_object("Examples/models/mnist_conv_clf.pkl")

################# EVALUATE HOWEVER YOU LIKE
def calculate_accuracy():
    test_loader = DataLoader(X_test, y_test, batch_size, shuffle=True)
    correct = 0
    for (X, y) in test_loader():
        y_pred = clf.predict(X)
        correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

    return (correct / len(y_test)) * 100

print(f"Accuracy: {calculate_accuracy()}%")

