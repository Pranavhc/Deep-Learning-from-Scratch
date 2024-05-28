import numpy as np
import pickle
from typing import Generator, Union

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle:bool= False):
        if y is not None and len(X) != len(y):
            raise ValueError("X and y must have the same length!")
        
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(y)
        self.samples = (self.len + batch_size - 1) // batch_size
    
    def __call__(self) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[np.ndarray, None, None]]:
        idx = np.arange(self.len)
        if self.shuffle: idx = np.random.permutation(idx)

        for i in range(0, self.len, self.batch_size):
            batch_idx = idx[i:i+self.batch_size]
            if self.y is not None:
                yield self.X[batch_idx], self.y[batch_idx]
            else: 
                yield self.X[batch_idx]

def shuffler(X, y):
    idx = np.arange(X.shape[0])
    idx = np.random.permutation(idx)
    return X[idx], y[idx]

def to_categorical1D(x: np.ndarray, n_col:Union[int, None]=None) -> np.ndarray:
    assert x.ndim == 1, "x should be 1-dimensional"

    if not n_col: n_col = np.max(x) + 1
    return np.eye(n_col)[x] # type: ignore

def save_object(model: object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

