import copy
import numpy as np

from .optim import Optimizer
from .regularization import Regularization
from typing import Union

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
    def __init__(self, input_size: int, output_size: int, regularization: Union[Regularization, None]=None) -> None:
        """ Initialize a dense layer."""
        self.input_size = input_size
        self.output_size = output_size
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
        """ ## backward pass
        1. Calculates gradients of loss with respect to parameters and input.
        2. Updates parameters.
        3. Applies regularization if specified.
        4. Returns gradient with respect to input.

        ### Mathematical description:
        Backpropagation calculates the gradient of the loss function:

            * Gradient of Weights: `∂L/∂w = ∂L/∂y_hat * ∂y_hat/∂w`
            * Gradient of Bias: `∂L/∂b = ∂L/∂y_hat`
            * Gradient of Input: `∂L/∂x = ∂L/∂y_hat * w`

        For an activation function, multiply its derivative with the loss derivative: `output_gradient = ∂L/∂y_hat * ∂A/∂y_hat`.
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
        return output_gradient * self.mask # sets gradient of the inactive neurons to zero

class Flatten(Layer):
    def __init__(self) -> None:
        self.prev_input_shape: tuple

    def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
        self.prev_input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient.reshape(self.prev_input_shape)

class Reshape(Layer):
    def __init__(self, output_shape):
        self.input_shape: tuple
        self.output_shape = output_shape

    def forward(self, input, train=True):
        self.input_shape = input.shape
        return input.reshape((self.input_shape[0], self.output_shape))

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)