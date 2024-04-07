import numpy as np
import pickle
from typing import Generator, Union

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle:bool= False):
        self.X, self.y = X, y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(y)
    
    def __call__(self) -> Union[Generator[tuple[np.ndarray, np.ndarray], None, None], Generator[np.ndarray, None, None]]:
        if self.y is not None: assert(len(self.X) == len(self.y)), "X and y must have the same length!"

        idx = np.arange(self.len)
        if self.shuffle: idx = np.random.permutation(idx)

        for i in range(0, self.len, self.batch_size):
            if self.y is not None:
                yield self.X[idx[i:i+self.batch_size]], self.y[idx[i:i+self.batch_size]]
            else: 
                yield self.X[idx[i:i+self.batch_size]]

def shuffler(X, y):
    idx = np.arange(X.shape[0]) # create an array from 0 to len(X)
    idx = np.random.permutation(idx) # permute the indexes
    return X[idx], y[idx] # return the input with permuted indexes

def to_categorical1D(x: np.ndarray, n_col:Union[int, None]=None) -> np.ndarray:
    assert x.ndim == 1, "x should be 1-dimensional"

    if not n_col: n_col = np.max(x) + 1
    return np.eye(n_col)[x]

def save_object(model: object, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_object(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

