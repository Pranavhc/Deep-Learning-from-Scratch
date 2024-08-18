import copy
import numpy as np

import nn.utils as utils
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
    """ Simple RNN layer"""
    def __init__(self, input_size:int, hidden_size:int, output_size:int, activation:Layer, regularization:Union[Regularization, None]=None) -> None:
        """ Args:
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
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation
        self.regularization = regularization

    def initialize(self, optimizer:Optimizer) -> None:
        """intialize the layer parameters"""
        self.input_weights = Dense(self.input_size, self.hidden_size, self.regularization)
        self.hidden_weights = Dense(self.hidden_size, self.hidden_size, self.regularization)
        self.output_weights = Dense(self.hidden_size, self.output_size, self.regularization)

        self.input_weights.initialize(optimizer)
        self.hidden_weights.initialize(optimizer)
        self.output_weights.initialize(optimizer)

    def forward(self, input_sequence:np.ndarray, hidden:np.ndarray|None=None, train:bool=True) -> tuple[np.ndarray, np.ndarray]:
        """ Args:
        input_sequence: `np.ndarray`
            Input sequence of shape *(batch_size, timesteps, input_size)*.
        hidden: `np.ndarray` | `None`
            Hidden state of shape *(batch_size, hidden_size)*.
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

class Conv2Dslow(Layer):
    """ 2D Convolutional layer (slow) <br>
    reference: [source of implementation](https://hackmd.io/@machine-learning/blog-post-cnnumpy-slow)
    """
    def __init__(self, in_chnls:int, out_chnls:int, kernel_size:int, stride:int=1, padding:int=0):
        """ Args:
        in_chnls: `int`
            Number of input channels.
        out_chnls: `int`
            Number of output channels.
        kernel_size: `int`
            Size of the kernel.
        stride: `int`
            Stride of the convolution.
        padding: `int`
            Padding of the input.
        """
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def initialize(self, optimizer: Optimizer) -> None:
        """intialize the layer parameters"""
        limit = 1 / np.sqrt(self.kernel_size * self.kernel_size)
        self.kernels = np.random.uniform(-limit, limit, size=(self.out_chnls, self.in_chnls, self.kernel_size, self.kernel_size))
        self.b = np.random.uniform(-limit, limit, size=(self.out_chnls))

        self.kernels_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def forward(self, input:np.ndarray, train:bool=True) -> np.ndarray:
        """ Args:
        input: `np.ndarray`
            Input of shape *(batch_size, channels, height, width)*.
        train: `bool`
            the mode of execution.
        
        Returns: `np.ndarray`: Output of the convolutional layer
        """
        self.input = input
        batch_size, in_chnls, in_height, in_width = input.shape

         # new dimension after convolution = ((in_dim + 2p - k) / s) + 1
        out_height = int((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1 
        out_width = int((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        out = np.zeros((batch_size, self.out_chnls, out_height, out_width))

        for i in range(batch_size):             # for each img
            for c in range(self.out_chnls):     # for each new channel
                for h in range(out_height):     # get kernel's vertical reach
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size

                    for w in range(out_width):  # get kernel's horizontal reach
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size

                        # fill each pixel in the output with convolution of input and kernel
                        out[i, c, h, w] = np.sum(input[i, :, h_start:h_end, w_start:w_end] * self.kernels[c, :,:,:]) + self.b[c]

        return out
    
    def backward(self, output_gradient:np.ndarray) -> np.ndarray:
        batch_size, in_chnls, in_height, in_width = self.input.shape
        batch_size, out_chnls, out_height, out_width = output_gradient.shape

        input_grad = np.zeros(self.input.shape)             # grad = conv(output_gradient, Kernels)
        kernels_grad = np.zeros(self.kernels.shape)         # grad = conv(output_gradient, input)
        bias_grad = np.sum(output_gradient, axis=(0, 2, 3)) # grad = sum(output_gradient)

        for i in range(batch_size):
            for c in range(out_chnls):
                for h in range(out_height):
                    h_start = h * self.stride
                    h_end = h_start + self.kernel_size
                    
                    for w in range(out_width):
                        w_start = w * self.stride
                        w_end = w_start + self.kernel_size
                                                
                        kernels_grad[c, :,:,:] += output_gradient[i, c, h, w] * self.input[i, :, h_start:h_end, w_start:w_end]
                        input_grad[i, :, h_start:h_end, w_start:w_end] += output_gradient[i, c, h, w] * self.kernels[c, :,:,:]

        # update parameters
        self.kernels = self.kernels_opt.update(self.kernels, kernels_grad)
        self.b = self.b_opt.update(self.b, bias_grad)

        return input_grad

class Conv2D(Layer):
    """ 2D Convolutional Layer <br>
    reference: [source of implementation](https://hackmd.io/@machine-learning/blog-post-cnnumpy-fast)
    """
    def __init__(self, in_chnls:int, out_chnls:int, kernel_size:int, stride:int=1, padding:int=0):
        """ Args:
        in_chnls: `int`
            Number of input channels.
        out_chnls: `int`
            Number of output channels.
        kernel_size: `int`
            Size of the kernel.
        stride: `int`
            Stride of the convolution.
        padding: `int`
            Padding of the input.
        """
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def initialize(self, optimizer: Optimizer) -> None:
        """intialize the layer parameters"""
        self.kernels = np.random.randn(self.out_chnls, self.in_chnls, self.kernel_size, self.kernel_size) * np.sqrt(1. / (self.kernel_size))
        self.b = np.random.randn(self.out_chnls) * np.sqrt(1. / (self.kernel_size))

        self.kernels_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def forward(self, input:np.ndarray, train:bool=True) -> np.ndarray:
        """ Args:
        input: `np.ndarray`
            Input of shape *(batch_size, channels, height, width)*.
        train: `bool`
            the mode of execution.
        
        Returns: `np.ndarray`: Output of the convolutional layer
        """
        self.input = input
        batch_size, in_chnls, in_height, in_width = input.shape

        out_height = int((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1
        out_width = int((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        input_cols = utils.img2col(input, self.kernel_size, self.kernel_size, self.stride, self.padding)
        kernel_cols = self.kernels.reshape((self.out_chnls, -1))
        b_col = self.b.reshape(-1, 1)

        # forward calculation & then reshape that matrix back to image
        out = kernel_cols @ input_cols + b_col
        out = np.array(np.hsplit(out, batch_size)).reshape((batch_size, self.out_chnls, out_height, out_width))

        self.input_cols, self.kernel_cols = input_cols, kernel_cols # save for backward pass
        return out
    
    def backward(self, output_gradient:np.ndarray) -> np.ndarray:
        input, input_cols, kernel_cols = self.input, self.input_cols, self.kernels
        batch_size, _, _, _ = input.shape

        out_grad = output_gradient.reshape(output_gradient.shape[0] * output_gradient.shape[1], output_gradient.shape[2] * output_gradient.shape[3])
        out_grad = np.array(np.vsplit(out_grad, batch_size))
        out_grad = np.concatenate(out_grad, axis=-1)

        input_grad_col = kernel_cols.T @ out_grad
        kernel_grad_col = out_grad @ input_cols.T

        input_grad = utils.col2img(input_grad_col, input.shape, self.kernel_size, self.kernel_size, self.stride, self.padding)
        kernel_grad = kernel_grad_col.reshape((kernel_cols.shape[0], self.in_chnls, self.kernel_size, self.kernel_size))
        self.b_grad = np.sum(output_gradient, axis=(0, 2, 3))

        self.kernels = self.kernels_opt.update(self.kernels, kernel_grad)
        self.b = self.b_opt.update(self.b, self.b_grad)

        return input_grad

class Dropout(Layer):
    """ Disables a fraction of neurons in the layer by setting them to zero. A common regularization technique."""
    def __init__(self, drop_rate:float=0.3) -> None:
        """ Args:
        drop_rate: `float`
            The probability of setting a neuron to zero.
        """
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
    """Attempts to flatten the incoming n-dimensional tensor into a 2D tensor of shape *(batch_size, -1)*"""
    def __init__(self) -> None:
        self.prev_input_shape: tuple

    def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
        self.prev_input_shape = input.shape
        return input.reshape(input.shape[0], -1)

    def backward(self, output_gradient: np.ndarray) -> np.ndarray:
        return output_gradient.reshape(self.prev_input_shape)

class Reshape(Layer):
    """Attempts to reshape the input into the specified shape."""
    def __init__(self, output_shape):
        """ Args:
        output_shape: `tuple`
            The desired shape (Exclude *batch size*)
        """
        self.input_shape: tuple
        self.output_shape = output_shape

    def forward(self, input, train=True):
        self.input_shape = input.shape
        return input.reshape((self.input_shape[0], self.output_shape))

    def backward(self, output_gradient):
        return output_gradient.reshape(self.input_shape)