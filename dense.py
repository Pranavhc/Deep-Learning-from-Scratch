from layer import Layer
from regularization import Regularization
import numpy as np

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, regularization: Regularization=None) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.regularization = regularization

    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        # calculate gradients
        weights_gradient = np.dot(output_gradient, self.input.T)
        input_gradient = np.dot(self.weights.T, output_gradient)
        
        # regularization
        if self.regularization:
            weights_gradient += self.regularization.regularize(self.weights)

        # update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient
