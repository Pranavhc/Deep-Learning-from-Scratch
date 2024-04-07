import numpy as np

class Optimizer():
    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class StochasticGradientDescent(Optimizer):
    """ SGD optimizer with momentum.
    
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

class RMSProp(Optimizer):
    """ RMSProp optimizer.
    
    Update formula: 
        ```python 
        c = decay_rate * c + (1 - decay_rate) * grad**2
        x = x - learning_rate * grad / (sqrt(c) + 1e-8)
        ```
    """
    def __init__(self, learning_rate: float = 0.01, decay_rate: float = 0.9) -> None:
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.cache = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.cache is None: self.cache = np.zeros(np.shape(parameters))

        self.cache =  self.decay_rate * self.cache + (1-self.decay_rate) * parameters_grad**2
        return parameters - self.learning_rate * parameters_grad / (np.sqrt(self.cache) + 1e-8)