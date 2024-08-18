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

#### Conv2D img2col and col2img functions

def get_indices(input_shape, kernel_height:int, kernel_width:int, stride:int, padding:int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """ Args: 
        input_shape: `tuple`
            input image shape *(batch_size, channels, height, width)*.
        kernel_height: `int`
            kernel height.
        kernel_width: `int`
            kernel width.
        stride: `int`
            stride value.
        padding: `int`
        
        Returns: `(np.ndarray, np.ndarray, np.ndarray)` <br>
            returns index matrices in order to transform an input image into a matrix. An helper function for `img2col` and `col2img` functions."""

        batch_size, in_chnls, in_height, in_width = input_shape

        # new dimension after convolution = ((in_dim + 2p - k) / s) + 1
        out_height = int((in_height + 2 * padding - kernel_height) / stride) + 1
        out_width = int((in_width + 2 * padding - kernel_width) / stride) + 1

        #### compute matrix of index i
        level1 = np.repeat(np.arange(kernel_height), kernel_width) # level 1 vector
        level1 = np.tile(level1, in_chnls) # stack level 1 vectors for each input channel
        # create a vector with an increase by 1 at each level.
        increasing_levels = stride * np.repeat(np.arange(out_height), out_width)
        i = level1.reshape(-1, 1) + increasing_levels.reshape(1, -1) # reshape both accordingly to get a matrix

        #### compute matrix of index j
        slide1 = np.tile(np.arange(kernel_width), kernel_height)
        slide1 = np.tile(slide1, in_chnls)
        increasing_slides = stride * np.tile(np.arange(out_width), out_height)
        j = slide1.reshape(-1, 1) + increasing_slides.reshape(1, -1) # reshape both accordingly to get a matrix

        chnl_delimitation = np.repeat(np.arange(in_chnls), kernel_height * kernel_width).reshape(-1, 1)

        return i, j, chnl_delimitation

def img2col(input:np.ndarray, kernel_height:int, kernel_width:int, stride:int, padding:int) -> np.ndarray:
    """ Args:
    input: `np.ndarray`
        input image.
    kernel_height: `int`
        kernel height.
    kernel_width: `int`
        kernel width.
    stride: `int`
        stride value.
    padding: `int`
        padding value.
        
    Returns:
        `np.ndarray` : matrix of input image after applying im2col operation.
    """
    input_padded = np.pad(input, ((0,0), (0,0), (padding, padding), (padding, padding)), mode='constant')
    i, j, d = get_indices(input.shape, kernel_height, kernel_width, stride, padding)

    cols = input_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols

def col2img(in_grad_col:np.ndarray, input_shape:tuple, kernel_height:int, kernel_width:int, stride:int, padding:int) -> np.ndarray:
    """ Args:
    in_grad_col: `np.ndarray`
        matrix of input gradients.
    input_shape: `tuple`
        input img shape *(batch_size, channels, height, width)*.
    kernel_height: `int`
        kernel height.
    kernel_width: `int`
        kernel width.
    stride: `int`
        stride value.
    pad: `int`
        padding value.
"""
    batch_size, in_chnls, in_height, in_width = input_shape

    h_pad, w_pad = in_height + 2 * padding, in_width + 2 * padding
    input_padded = np.zeros((batch_size, in_chnls, h_pad, w_pad))

    i, j, chnl_delimitation = get_indices(input_shape, kernel_height, kernel_width, stride, padding)
    
    in_grad_col_reshaped = np.array(np.hsplit(in_grad_col.reshape(-1, in_grad_col.shape[-1]), batch_size))
    np.add.at(input_padded, (slice(None), chnl_delimitation, i, j), in_grad_col_reshaped) # type: ignore

    if padding == 0: return input_padded
    else: return input_padded[padding:-padding, padding:-padding, :, :]