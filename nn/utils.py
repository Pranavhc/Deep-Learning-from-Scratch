import numpy as np
from typing import Generator

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(y)
    
    def __call__(self) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        if self.y is not None: assert(len(self.X) == len(self.y)), "X and y must have the same length!"

        idx = np.arange(self.len)
        if self.shuffle: idx = np.random.permutation(idx)

        for i in range(0, self.len, self.batch_size):
            if self.y is not None:
                yield self.X[idx[i:i+self.batch_size]], self.y[idx[i:i+self.batch_size]]
            else: 
                yield self.X[idx[i:i+self.batch_size]]

def to_categorical1D(x: np.ndarray, n_col:int=None) -> np.ndarray:
    assert x.ndim == 1, "x should be 1-dimensional"

    if not n_col: n_col = np.max(x) + 1
    return np.eye(n_col)[x]
