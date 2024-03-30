import numpy as np
from .layers import Layer

class ReLu(Layer):
  """ Rectified Linear Unit (ReLU) activation function. """
  def forward(self, input: np.ndarray) -> np.ndarray:
    """ forward pass. Returns the output of the ReLU function."""
    self.input = input
    return np.maximum(0, self.input)

  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    """ backward pass. Chains the output gradient to the input gradient. """
    grad = np.where(self.input < 0, 0, 1)
    return np.multiply(output_gradient, grad)

class Sigmoid(Layer):
  """ Sigmoid activation function. """
  def forward(self, input: np.ndarray) -> np.ndarray:
    """ forward pass. Returns the output of the ReLU function."""
    self.input = input
    return 1/(1 + np.exp(-self.input))

  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    """ backward pass. Chains the output gradient to the input gradient. """
    sigmoid = self.forward(self.input)
    grad = sigmoid * (1 - sigmoid)
    return np.multiply(output_gradient, grad)

class Tanhh(Layer):
  """ Tanh activation function. """
  def forward(self, input: np.ndarray) -> np.ndarray:
    """ forward pass. Returns the output of the ReLU function."""
    self.input = input
    return np.tanh(self.input)
  
  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    """ backward pass. Chains the output gradient to the input gradient. """
    return output_gradient * (1 - np.tanh(self.input) ** 2)

class Softmax(Layer):
  """ Softmax activation function. """
  def forward(self, input: np.ndarray) -> np.ndarray:
    """ forward pass. Returns the output of the Softamax function."""
    self.input = input
    e_x = np.exp(input - np.max(input, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
  
  def backward(self, output_gradient: np.ndarray) -> np.ndarray:
    """ backward pass. Chains the output gradient to the input gradient. """
    softmax = self.forward(self.input)
    grad = np.multiply(output_gradient, softmax)
    grad = grad - np.sum(grad, axis=-1, keepdims=True) * softmax
    return grad