import numpy as np

class Optimizer():
    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class StochasticGradientDescent(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.velocity is None: self.velocity = np.zeros(np.shape(parameters))

        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * parameters_grad
        return parameters - self.learning_rate * self.velocity
    