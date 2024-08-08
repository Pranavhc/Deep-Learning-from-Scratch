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
        """ backward pass
        1. Calculates gradients of loss with respect to parameters and input.
        2. Updates parameters.
        3. Applies regularization if specified.
        4. Returns gradient with respect to input.

        Mathematical description:

            Gradient of Weights: `∂L/∂w = ∂L/∂y_hat * ∂y_hat/∂w`
            Gradient of Bias: `∂L/∂b = ∂L/∂y_hat`
            Gradient of Input: `∂L/∂x = ∂L/∂y_hat * w`
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

class RNN(Layer):
    """ Simple RNN layer:
    input_size: `int`
        Number of input features.
    hidden_size: `int`
        Number of hidden units.
    output_size: `int`
        Number of output features.
    activation: `Layer`
        Activation function to use.
    regularization: `Regularization` | `None`
        Regularization to use.
    """
    def __init__(self, input_size:int, hidden_size:int, output_size:int, activation:Layer, regularization:Union[Regularization, None]=None) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.regularization = regularization

    def initialize(self, optimizer:Optimizer) -> None:
        self.input_weights = Dense(self.input_size, self.hidden_size, self.regularization)
        self.hidden_weights = Dense(self.hidden_size, self.hidden_size, self.regularization)
        self.output_weights = Dense(self.hidden_size, self.output_size, self.regularization)

        self.input_weights.initialize(optimizer)
        self.hidden_weights.initialize(optimizer)
        self.output_weights.initialize(optimizer)

    def forward(self, input_sequence:np.ndarray, hidden:np.ndarray=None, train:bool=True) -> tuple[np.ndarray, np.ndarray]:
        """ Args:
        input_sequence: `np.ndarray`
            Input sequence of shape (batch_size, timesteps, input_size).
        hidden: `np.ndarray`
            Hidden state of shape (batch_size, hidden_size).
        train: `bool`
            the mode of execution.
        """
        batch_size, timesteps, _ = input_sequence.shape
        if hidden is None: hidden = np.zeros((batch_size, self.hidden_size))

        outputs = []
        for t in range(timesteps):
            input_t = input_sequence[:, t, :]
            combined_hidden_and_input = self.input_weights.forward(input_t) + self.hidden_weights.forward(hidden)
            hidden = self.activation.forward(combined_hidden_and_input)
            output = self.output_weights.forward(hidden)
            outputs.append(output)

        outputs = np.stack(outputs, axis=1)
        return outputs, hidden

    def backward(self, output_gradient:np.ndarray) -> np.ndarray:
        """ Args:
        output_gradient: `np.ndarray`   
            Gradient of the loss w.r.t. the output of the RNN layer. Shape (batch_size, timesteps, output_size).
        """
        timesteps = output_gradient.shape[1]
        grad_hidden = np.zeros((output_gradient.shape[0], self.hidden_size))
        
        for t in reversed(range(timesteps)):
            grad_output = output_gradient[:, t, :]
            grad = self.output_weights.backward(grad_output)
            grad += grad_hidden
            grad = self.activation.backward(grad)
            grad_hidden = self.hidden_weights.backward(grad)
            self.input_weights.backward(grad)

        return grad_hidden

class Dropout(Layer):
    """ Dropout layer:
    drop_rate: `float`
        The probability of setting a neuron to zero.
    """
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