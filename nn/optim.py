import numpy as np

class Optimizer():
    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# https://en.wikipedia.org/wiki/Stochastic_gradient_descent#Momentum
class StochasticGradientDescent(Optimizer):
    """ SGD with momentum.
    
    Update formula: 
        ```python 
        v = m * v + (1 - m) * grad
        x = x - learning_rate * v
        ```
    """
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0) -> None:
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.velocity is None: self.velocity = np.zeros(np.shape(parameters))

        self.velocity = self.momentum * self.velocity + (1 - self.momentum) * parameters_grad
        return parameters - self.learning_rate * self.velocity

# https://en.wikipedia.org/wiki/Stochastic_gradient_descent#AdaGrad
class AdaGrad(Optimizer):
    """ AdaGrad optimizer.
    
    Update formula: 
        ```python 
        c = c + grad**2
        x = x - learning_rate * grad / (sqrt(c) + 1e-8)
        ```
    """
    def __init__(self, learning_rate: float = 0.01) -> None:
        self.learning_rate = learning_rate
        self.cache = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.cache is None: self.cache = np.zeros(np.shape(parameters))

        self.cache = self.cache + parameters_grad**2
        return parameters - self.learning_rate * parameters_grad / (np.sqrt(self.cache) + 1e-8)