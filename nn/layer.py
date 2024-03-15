import numpy as np

class Layer:
    def __init__(self) -> None:
        self.input = None
        self.output = None

    def forward(self, input: np.ndarray) -> np.ndarray:
        pass # return the linear output

    def backward(self, output_gradient: np.ndarray, learning_rate: float) -> np.ndarray:
        pass # return the input gradient