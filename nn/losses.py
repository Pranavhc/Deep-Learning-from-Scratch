import numpy as np

class Loss:
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        raise NotImplementedError

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        raise NotImplementedError

# Mean Squared Error
class MSE(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.power(y_true - y_pred, 2))

    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / np.size(y_true)
    

# Catagorical Cross Entropy
class CrossEntropy(Loss):
    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred)
    
    def grad(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -(y_true / y_pred) + (1 - y_true) / (1 - y_pred)
        