import numpy as np
from .layer import Layer

class ReLu(Layer):
  def forward(self, input: np.ndarray) -> np.ndarray:
    self.input = input
    return np.maximum(0, self.input)

  def backward(self, output_gradient: np.ndarray, lr: float) -> np.ndarray:
    grad = np.where(self.input < 0, 0, 1)
    return np.multiply(output_gradient, grad)

class Sigmoid(Layer):
  def forward(self, input: np.ndarray) -> np.ndarray:
    self.input = input
    return 1/(1 + np.exp(-self.input))

  def backward(self, output_gradient: np.ndarray, lr: float) -> np.ndarray:
    sigmoid = self.forward(self.input)
    grad = sigmoid * (1 - sigmoid)
    return np.multiply(output_gradient, grad)

class Tanhh(Layer):
    def forward(self, input) -> np.ndarray:
        self.input = input
        return np.tanh(self.input)
    
    def backward(self, output_gradient: np.ndarray, lr: float) -> np.ndarray:
        return output_gradient * (1 - np.tanh(self.input) ** 2)

class Softmax(Layer):
    def forward(self, input) -> np.ndarray:
        exp_z = np.exp(input)
        self.output = exp_z / np.sum(exp_z)
        return self.output
    
        
    def backward(self, output_gradient: np.ndarray, lr: float) -> np.ndarray:
        n = np.size(self.output)
        return np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
