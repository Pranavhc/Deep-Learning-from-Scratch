import numpy as np
from .layers import Layer

class ReLu(Layer):
  """ Rectified Linear Unit (ReLU) activation function. """
  def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
    self.input = input
    return np.maximum(0, self.input)

  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    grad = np.where(self.input < 0, 0, 1)
    return np.multiply(output_gradient, grad)

class Sigmoid(Layer):
  """ Sigmoid activation function. """
  def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
    self.input = input
    return 1/(1 + np.exp(-self.input))

  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    sigmoid = self.forward(self.input)
    grad = sigmoid * (1 - sigmoid)
    return np.multiply(output_gradient, grad)

class Tanh(Layer):
  """ Tanh activation function. """
  def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
    self.input = input
    return np.tanh(self.input)
  
  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    return output_gradient * (1 - np.tanh(self.input) ** 2)

class Softmax(Layer):
  """ Softmax activation function. """
  def forward(self, input: np.ndarray, train:bool=True) -> np.ndarray:
    self.input = input
    e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
  
  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    softmax = self.forward(self.input)
    grad = np.multiply(output_gradient, softmax)
    grad = grad - np.sum(grad, axis=-1, keepdims=True) * softmax
    return grad