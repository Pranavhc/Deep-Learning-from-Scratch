import numpy as np

class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input) -> np.ndarray:
        pass

    def backward(self, output_gradient, learning_rate) -> np.ndarray:
        # update parameters and return the input gradient
        pass