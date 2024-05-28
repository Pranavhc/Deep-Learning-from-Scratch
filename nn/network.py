from typing import Union
import numpy as np
from tqdm import tqdm

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
        loss_grad = self.loss.grad(y, y_pred)

        self._backward(loss_grad)
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

    def fit(self, train_data: DataLoader, val_data: Union[DataLoader, None]=None, epochs:int=10, accuracy:bool=False, verbose:bool=True) -> Union[tuple[dict, dict], dict]:
        for e in range(epochs):
            train_batch_loss, val_batch_loss = [], []
            train_batch_acc, val_batch_acc = [], []

            with tqdm(train_data(), unit='batch', disable=not verbose) as pbar:
                for X_train, y_train in pbar:
                    pbar.set_description(f"Epoch {e+1}/{epochs}")
                    pbar.total = train_data.samples
                    pbar.bar_format = "{l_bar}{bar:20}| {n_fmt}/{total_fmt}{postfix}"
                

                    train_loss, y_pred_train = self._train_on_batch(X_train, y_train)
                    train_batch_loss.append(train_loss)
                    
                    if accuracy: train_batch_acc.append(self.acc(y_pred_train, y_train))
                
                    if accuracy: pbar.set_postfix(accuracy=f"{np.mean(train_batch_acc):.4f}", loss=f"{np.mean(train_batch_loss):.4f}")
                    else: pbar.set_postfix(loss=f"{np.mean(train_batch_loss):.4f}")

                self.error['train'].append(float(np.mean(train_batch_loss)))
                if accuracy: self.accuracy['train'].append(np.mean(train_batch_acc))

            if val_data:                
                for X_val, y_val in val_data():
                    val_loss, y_pred_val = self._test_on_batch(X_val, y_val)
                    val_batch_loss.append(val_loss)
                    
                    if accuracy: val_batch_acc.append(self.acc(y_pred_val, y_val))
            
                if accuracy: tqdm.write(f"val_accuracy={np.mean(val_batch_acc):.4f}, val_loss={np.mean(val_batch_loss):.4f} \n")
                else: tqdm.write(f"val_loss={np.mean(val_batch_loss):.4f} \n")

                self.error['val'].append(float(np.mean(val_batch_loss)))
                if accuracy: self.accuracy['val'].append(np.mean(val_batch_acc))
        
        if accuracy: return self.error, self.accuracy
        return self.error
