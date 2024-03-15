import numpy as np

class Loss:
    def __init__(self, loss_fn, loss_prime):
        self.loss_fn = loss_fn
        self.loss_prime = loss_prime

    def __call__(self, y_true, y_pred):
        return self.loss_fn(y_true, y_pred)

    def prime(self, y_true, y_pred):
        return self.loss_prime(y_true, y_pred)

# Mean Squared Error
class MSE(Loss):
    def __init__(self):
        def mse(y_true, y_pred):
            return np.mean(np.power(y_true - y_pred, 2))

        def mse_prime(y_true, y_pred):
            return 2 * (y_pred - y_true) / np.size(y_true)
        
        super().__init__(mse, mse_prime)

# Catagorical Cross Entropy
class CrossEntropy(Loss):
    def __init__(self):
        def cross_entropy(y_true, y_pred):
            # these couple quirks keeps y_pred from causing a divide by zero error.
            epsilon = 1e-15  # Small constant
            y_pred = np.clip(y_pred, epsilon, 1 - epsilon) # keeps the values between epsilon and 1-epsilon
            return -np.sum(y_true * np.log(y_pred)) / len(y_true)
        
        def cross_entropy_prime(y_true, y_pred):
            return -y_true / y_pred 
        
        super().__init__(cross_entropy, cross_entropy_prime)