import numpy as np

class Optimizer():
    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.accum_w_updt = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.accum_w_updt is None: self.accum_w_updt = np.zeros(np.shape(parameters))

        self.accum_w_updt = self.momentum * self.accum_w_updt + (1 - self.momentum) * parameters_grad
        return parameters - self.learning_rate * self.accum_w_updt
    