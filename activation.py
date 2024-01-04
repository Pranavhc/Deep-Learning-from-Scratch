import numpy as np
from layer import Layer

class Activation(Layer):
    def __init__(self, activation, activation_prime) -> None:
        self.activation = activation
        self.activation_prime = activation_prime

    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient, self.activation_prime(self.input))

class ReLu(Activation):
    def __init__(self) -> None:
        def relu(x):
            return np.maximum(0, x)
        
        def relu_prime(x):
            return np.where(x < 0, 0, 1)
        
        super().__init__(relu, relu_prime)

class Sigmoid(Activation):
    def __init__(self) -> None:
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))

        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)

        super().__init__(sigmoid, sigmoid_prime)

class Tanh(Activation):
    def __init__(self) -> None:
        def tanh(x):
            return np.tanh(x)
        
        def tanh_prime(x):
            return 1 - np.tanh(x) ** 2

        super().__init__(tanh, tanh_prime)
