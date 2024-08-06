# We usually add a regularization term to the loss function to prevent large weights.
# The term includes the magnitude of the weights (the norm of weights) and a hyperparameter lambda to tweak the regularization strength.
# By adding the magnitude of the weights to the loss function, we penalize large weights and prevent overfitting.

# Here, we achieve the same effect by calculating the derivative of the regularization term w.r.t the weights and adding it to the weights' gradients.

# "L1 norm" is also known as "Lasso" regularization, and "L2 norm" is known as "Ridge" regularization.
# "L1 norm" for vectors is the Manhattan norm, and "L2 norm" is the Euclidean norm.
# For tensors, "L2 norm" is known as the Frobenius norm. But here, we are dealing with vectors.

import numpy as np

class Regularization():
    """
    Returns the derivative of regularization term applied on weights.

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
        