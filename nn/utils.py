import numpy as np
from typing import Generator, Tuple

<<<<<<< HEAD:nn/utils.py
# this doesn't work yet (this function works but not with the current implementation of the network)
# i don't know how to feed a batch to the network, i'm running into all sorts of shape errors
=======
# This doesn't work yet (the function works but not with the current implementation of the network)
# I don't know how to feed a batch to the network, I'm running into all sorts of shape errors
>>>>>>> f2f50e2b9638ed611731ef463733068969586b60:utils.py

class DataLoader:
    def __init__(self, X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: False):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.len = len(y)
    
    def __call__(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        assert(len(self.X) == len(self.y)), "X and y must have the same length!"

        idx = np.arange(self.len)
        if self.shuffle: idx = np.random.permutation(idx)

        for i in range(0, self.len, self.batch_size):
            yield self.X[idx[i:i+self.batch_size]], self.y[idx[i:i+self.batch_size]]
