import copy
import numpy as np

from .optim import Optimizer
from .regularization import Regularization

class Layer:
    """The interface that each layer should implement."""
    def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
        """forward pass. Returns the output of the layer."""
        raise NotImplementedError

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """backward pass. Returns the gradient with respect to the input of the layer."""
        raise NotImplementedError

class Dense(Layer):
    """A dense (fully connected) layer in a neural network."""

    def __init__(self, input_size: int, output_size: int, regularization: Regularization=None) -> None:
        """ Initialize a dense layer."""

        self.input = None 
        self.input_size = input_size
        self.output_size = output_size

        self.weights = None
        self.bias = None

        self.regularization = regularization

    def initialize(self, optimizer: Optimizer) -> None:
        """intialize the layer parameters"""

        limit = 1/np.sqrt(self.input_size)
        
        self.weights = np.random.uniform(-limit, limit, size=(self.input_size, self.output_size))
        self.bias = np.random.uniform(-limit, limit, size=(1, self.output_size))

        # save state of the optimizer for parameters of this layer
        self.W_opt = copy.copy(optimizer) 
        self.b_opt = copy.copy(optimizer) 

    def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
        """forward pass. Returns input.dot(weights) + bias"""
        self.input = input
        return np.dot(self.input, self.weights) + self.bias
    
    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        """ 
        ## backward pass
        1. Calculates grdients of loss wrt parameters and input of the layer.
        2. Updates parameters. 
        3. Applies regularization if given. 
        4. Returns gradient of input of the layer.
        
        ### Mathematical description:
        Backpropagation - the process of calculating the gradient of the loss function

            if `∂L/∂y_hat = 2*(y_hat-y)/n` then

            Gradient of Weights: `∂L/∂w = ∂L/∂y_hat * ∂y_hat/∂w = 2*(y_hat-y)/n * x`

            Gradient of Bias: `∂L/∂b = ∂L/∂y_hat * ∂y_hat/∂b = 2*(y_hat-y)/n`

            Gradient of Input: `∂L/∂x = ∂L/∂y_hat * ∂y_hat/∂x = 2*(y_hat-y)/n * w`

        By chain rule, if we take the activation function into account, we are supposed to multiply its
        derivative with the derivative of loss w.r.t to y_hat, I.e. `output_gradient = ∂L/∂y_hat * ∂A/∂y_hat`.
        """

        # calculate gradients                                       # L = loss  Y = output  W = weights  X = input
        weights_gradient = np.dot(self.input.T, output_gradient)    # ∂L/dW = ∂L/dY * ∂Y/∂W = output_gradient * input
        bias_grad = np.sum(output_gradient, axis=0, keepdims=True)  # ∂L/db = ∂L/dY * ∂Y/∂b = output_gradient
        
        input_gradient = np.dot(output_gradient, self.weights.T)    # ∂L/dX = ∂L/dY * ∂Y/∂X = output_gradient * weights
        
        # regularization
        if self.regularization:
            weights_gradient += self.regularization(self.weights)

        # update parameters
        self.weights = self.W_opt.update(self.weights, weights_gradient)
        self.bias = self.b_opt.update(self.bias, bias_grad)
        
        return input_gradient

class Dropout(Layer):
    def __init__(self, drop_rate:float=0.3) -> None:
        self.drop_rate = drop_rate
        self.mask = None

    def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
        self.input = input
        self.mask = (1 - self.drop_rate)
        
        if train: self.mask = np.random.choice([0, 1], size=input.shape, p=[self.drop_rate, 1-self.drop_rate])
        return input * self.mask

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient * self.mask
