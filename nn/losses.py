import numpy as np

class Loss:
    """All loss functions must implement this interface."""
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the loss."""
        raise NotImplementedError

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the loss with respect to the predicted labels."""
        raise NotImplementedError

# Mean Squared Error
class MSE(Loss):
    """Mean Squared Error Loss Function.
    Computes the mean squared error between the true labels and the predicted ones.
    """

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute the MSE loss."""
        return np.mean(np.power(y_true - y_pred, 2))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Compute the gradient of the MSE loss with respect to the predicted labels."""
        return 2 * (y_pred - y_true) / np.size(y_true)
    

# Binary Cross Entropy
class BinaryCrossEntropy(Loss):
    """Binary CrossEntropy Loss Function.
    Computes the cross entropy between the true labels and the predicted ones.
    """
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)