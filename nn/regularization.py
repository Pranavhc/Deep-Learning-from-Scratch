import numpy as np

class Regularization():
    """
    Returns the derivative of regularization norm applied on weights.

    regularization norms
    ---
    - L1 = lamda * sum(|W|)
    - L2 = lambda * Sum(W^2)

    returns
    ---
    The Derivative of either L1 or L2 norm w.r.t weights.
    - L1 = lambda * sign(W)
    - L2 = 2 * lambda * W
    """
    def __init__(self, norm: str, lambda_: float ):
        if norm not in ["L1", "L2"]: raise ValueError("Invalid norm!")
        
        self.lambda_ = lambda_
        self.norm = norm

    def __call__(self, weights: np.ndarray) -> np.ndarray:
        if self.norm == "L1": return self.lambda_ * np.sign(weights)
        return 2 * self.lambda_ * weights # for 'L2' 
        