import numpy as np

class Regularization():
    def __init__(self, term: str, lambda_: float ):
        self.lambda_ = lambda_
        self.term = term

    def L1(self, weights: np.ndarray) -> np.ndarray:
        return self.lambda_ * np.sign(weights)
    
    def L2(self, weights: np.ndarray) -> np.ndarray:
        return 2 * self.lambda_ * weights

    def regularize(self, weights: np.ndarray) -> np.ndarray:
        if self.term == "L1":
            return self.L1(weights)
        return self.L2(weights)

# Regularization Terms -
# L1 = lamda * sum(|W|)
# L2 = lambda * Sum(W^2)

# Derivative w.r.t W -
# L1 = lambda * sign(W)
# L2 = 2 * lambda * W