import numpy as np

class Regularization():
    def __init__(self, norm: str, lambda_: float ):
        self.lambda_ = lambda_
        self.norm = norm

    def L1(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_ * np.sign(weights)
    
    def L2(self, weights: np.ndarray) -> np.ndarray:
        return 2 * self.lambda_ * weights

    def regularize(self, weights: np.ndarray) -> np.ndarray:
        if self.norm == "L1":
            return self.L1(weights)
        return self.L2(weights)