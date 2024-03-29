import numpy as np

from .layers import Layer
from .losses import Loss
from .optim import Optimizer
from .utils import DataLoader


class NeuralNetwork:
    def __init__(self, optimizer: Optimizer, loss: Loss, layers: list[Layer]) -> None:
        self.optimizer = optimizer
        self.loss = loss
        self.layers = layers

        self.error = {'train': [], 'val': []}

        for layer in layers:
            if hasattr(layer, 'initialize'):
                layer.initialize(optimizer=self.optimizer)

    def _forward(self, input: np.ndarray) -> np.ndarray: 
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def _backward(self, loss_grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def _train_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self._forward(X)
        loss = np.mean(self.loss(y, y_pred))
        self._backward(self.loss.grad(y, y_pred))
        return loss
    
    def _test_on_batch(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self._forward(X)
        loss = np.mean(self.loss(y, y_pred))
        return loss

    def predict(self, input):
        return self._forward(input)

    def fit(self, train_data: DataLoader, val_data: DataLoader=None, epochs:int=10, verbose:bool=True) -> tuple[list[float], list[float]]:
        for e in range(epochs):
            train_batch_loss = []
            val_batch_loss = []
            for X_train, y_train in train_data():
                train_loss = self._train_on_batch(X_train, y_train)
                train_batch_loss.append(train_loss)
            self.error['train'].append(np.mean(train_batch_loss))

            if val_data:
                for X_val, y_val in val_data():
                    val_loss = self._test_on_batch(X_val, y_val)
                    val_batch_loss.append(val_loss)
                self.error['val'].append(np.mean(val_batch_loss))

            if verbose:
                if val_data: print(f"{e}/{epochs}\t- train loss: {self.error['train'][-1]}\t- val loss: {self.error['val'][-1]}")
                else: print(f"{e}/{epochs}\t- train loss: {np.mean(self.error['train'][-1])}")
        
        return (self.error['train'], self.error['val'])