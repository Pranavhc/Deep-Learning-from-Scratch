from typing import Union
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
        self.accuracy = {'train': [], 'val': []}

        for layer in layers:
            if hasattr(layer, 'initialize'): layer.initialize(optimizer=self.optimizer) # type: ignore

    def _forward(self, input: np.ndarray, train:bool) -> np.ndarray: 
        output = input
        for layer in self.layers:
            output = layer.forward(output, train=train)
        return output
    
    def _backward(self, loss_grad: np.ndarray) -> None:
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def _train_on_batch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        y_pred = self._forward(X, train=True)
    
        try: loss = np.mean(self.loss(y, y_pred))
        except: loss = self.loss(y, y_pred)

        self._backward(self.loss.grad(y, y_pred))
        return float(loss), y_pred
    
    def _test_on_batch(self, X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
        y_pred = self._forward(X, train=False)
    
        try: loss = np.mean(self.loss(y, y_pred))
        except: loss = self.loss(y, y_pred)

        return float(loss), y_pred

    def acc(self, y_pred: np.ndarray, y:np.ndarray) -> float:
        assert y_pred.shape == y.shape, "Shapes of y_pred and y must be the same"

        if y_pred.ndim >= 1 and y.ndim >= 1:
            return np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)) / len(y) * 100
        else: return np.sum(y_pred == y) / len(y) * 100

    def predict(self, input):
        return self._forward(input, train=False)

    def fit(self, train_data: DataLoader, val_data: Union[DataLoader, None]=None, epochs:int=10, verbose:bool=True, accuracy:bool=False) -> Union[tuple[dict, dict], dict]:
        for e in range(epochs):
            train_batch_loss, val_batch_loss = [], []
            train_batch_acc, val_batch_acc = [], []

            for X_train, y_train in train_data():
                train_loss, y_pred_train = self._train_on_batch(X_train, y_train)
                train_batch_loss.append(train_loss)
                if accuracy: train_batch_acc.append(self.acc(y_pred_train, y_train))
            self.error['train'].append(float(np.mean(train_batch_loss)))
            if accuracy: self.accuracy['train'].append(np.mean(train_batch_acc))

            if val_data:
                for X_val, y_val in val_data():
                    val_loss, y_pred_val = self._test_on_batch(X_val, y_val)
                    val_batch_loss.append(val_loss)
                    if accuracy: val_batch_acc.append(self.acc(y_pred_val, y_val))
                self.error['val'].append(float(np.mean(val_batch_loss)))
                if accuracy: self.accuracy['val'].append(np.mean(val_batch_acc))

            if verbose:
                if val_data and not accuracy: print(f" {e}/{epochs}\t- loss: {self.error['train'][-1]:.4f}  -  val_loss: {self.error['val'][-1]:.4f}")
                elif val_data and accuracy: print(f" {e}/{epochs}\t- loss: {self.error['train'][-1]:.4f}  -  accuracy: {self.accuracy['train'][-1]:.4f}  -  val_loss: {self.error['val'][-1]:.4f}  -  val_accuracy: {self.accuracy['val'][-1]:.4f}")
                elif not val_data and accuracy: print(f" {e}/{epochs}\t- loss: {self.error['train'][-1]:.4f}  -  accuracy: {self.accuracy['train'][-1]:.4f}")
                else: print(f" {e}/{epochs}\t- loss: {self.error['train'][-1]:.4f}")
        
        if accuracy: return self.error, self.accuracy
        return self.error
