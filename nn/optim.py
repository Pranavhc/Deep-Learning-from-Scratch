import numpy as np

class Optimizer():
    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class SGD(Optimizer):
    """ ### SGD optimizer with momentum
    
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
    """ ### AdaGrad optimizer
    
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
    """ ### RMSProp optimizer
    
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
    

class Adam(Optimizer):
    """ ### Adam optimizer
    
    Update formula: 
        ```python 
        momentum = decay_rate_1 * m + (1 - decay_rate_1) * grad
        velocity = decay_rate_2 * v + (1 - decay_rate_2) * grad**2

        m = momentum / (1 - decay_rate_1)
        v = velocity / (1 - decay_rate_2)

        x = x - learning_rate * m / (sqrt(v) + 1e-8)
        ```
    """
    def __init__(self, learning_rate:float = 0.01, b1:float=0.9, b2:float=0.995 ) -> None:
        self.learning_rate = learning_rate
        self.b1 = b1
        self.b2 = b2

        self.momentum = None
        self.velocity = None

    def update(self, parameters: np.ndarray, parameters_grad: np.ndarray) -> np.ndarray:
        if self.momentum is None:
            self.momentum = np.zeros(np.shape(parameters))
            self.velocity = np.zeros(np.shape(parameters))

        
        self.momentum = self.b1 * self.momentum + (1 - self.b1) * parameters_grad
        self.velocity = self.b2 * self.velocity + (1 - self.b2) * parameters_grad**2

        # since momentum and velocity are zero intially, they tend to be close to zero.
        # It's a bias that we have in the first few iterations.
        # To avoid this bias, we scale up the momentum and velocity.
        m = self.momentum / (1 - self.b1)
        v = self.velocity / (1 - self.b2)

        return parameters - self.learning_rate * m / (np.sqrt(v) + 1e-8)