from layer import Layer
from regularization import Regularization
import numpy as np

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int, regularization: Regularization=None) -> None:
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.randn(output_size, 1)
        self.regularization = regularization

    def forward(self, input: np.ndarray) -> np.ndarray:
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        # calculate gradients
        weights_gradient = np.dot(output_gradient, self.input.T) # ∂L/dW = ∂L/dY * ∂Y/∂W = output_gradient * input
        input_gradient = np.dot(self.weights.T, output_gradient) # ∂L/dX = ∂L/dY * ∂Y/∂X = output_gradient * weights
        
        # regularization
        if self.regularization:
            weights_gradient += self.regularization.regularize(self.weights)

        # update parameters
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        return input_gradient



# Backpropagation - the process of calculating the gradient of the loss function

# ∂L/∂y_hat = 2*(y_hat-y)/n
# ∂L/∂w = ∂L/∂y_hat * ∂y_hat/∂w = 2*(y_hat-y)/n * x
# ∂L/∂b = ∂L/∂y_hat * ∂y_hat/∂b = 2*(y_hat-y)/n
# ∂L/∂x = ∂L/∂y_hat * ∂y_hat/∂x = 2*(y_hat-y)/n * w

# By chain rule, if we take activation into account, we are supposed to multiply the 
# derivative of loss w.r.t to y_hat with the derivative of activation w.r.t to y_hat