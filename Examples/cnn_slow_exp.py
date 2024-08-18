import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import correlate

from nn.network import NeuralNetwork
from nn.optim import Adam, Optimizer
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Layer, Dense, Flatten
from nn.activations import ReLu, Softmax
from nn.utils import DataLoader, to_categorical1D, shuffler, save_object, load_object

class Conv2Dexperimental(Layer):
    def __init__(self, in_chnls:int, out_chnls:int, kernel_size:int, stride:int=1, padding:int=0):
        self.in_chnls = in_chnls
        self.out_chnls = out_chnls
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def initialize(self, optimizer: Optimizer) -> None:
        limit = 1 / np.sqrt(self.kernel_size * self.kernel_size)
        self.kernels = np.random.uniform(-limit, limit, size=(self.out_chnls, self.in_chnls, self.kernel_size, self.kernel_size))
        self.b = np.random.uniform(-limit, limit, size=(self.out_chnls))

        self.kernels_opt = copy.copy(optimizer)
        self.b_opt = copy.copy(optimizer)

    def forward(self, input:np.ndarray, train:bool=True) -> np.ndarray:
        self.input = input
        batch_size, in_chnls, in_height, in_width = input.shape

         # new dimension after convolution = ((in_dim + 2p - k) / s) + 1
        out_height = int((in_height + 2 * self.padding - self.kernel_size) / self.stride) + 1 
        out_width = int((in_width + 2 * self.padding - self.kernel_size) / self.stride) + 1

        out = np.zeros((batch_size, self.out_chnls, out_height, out_width))

        if self.padding > 0:
            input = np.pad(input, ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)), mode='constant')

        for i in range(batch_size):             # for each img
            for c in range(self.out_chnls):     # for each new channel
                convolved = correlate(input[i], self.kernels[c], mode='valid')
                strided_conv = convolved[:, ::self.stride, ::self.stride]
                out[i, c, :,:] = strided_conv + self.b[c]

        return out
    
    def backward(self, output_gradient:np.ndarray) -> np.ndarray:
        batch_size, in_chnls, in_height, in_width = self.input.shape
        batch_size, out_chnls, out_height, out_width = output_gradient.shape

        input_grad = np.zeros(self.input.shape)             
        kernels_grad = np.zeros(self.kernels.shape)         
        bias_grad = np.sum(output_gradient, axis=(0, 2, 3)) 

        padded_input = np.pad(self.input, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1), (self.kernel_size - 1, self.kernel_size - 1)), mode='constant')

        for i in range(batch_size):
            for c in range(out_chnls):
                for k in range(in_chnls):
                    # Calculate kernel gradient
                    # print(f"\nShapes: Kernel: {kernels_grad[c, k].shape} Input: {self.input[i, k].shape}, Grad: {output_gradient[i, c].shape}")
                    corr_result = correlate(padded_input[i, k], output_gradient[i, c], mode='valid')
                    kernels_grad[c, k] += corr_result[:self.kernel_size, :self.kernel_size]
                    
                    # # Calculate input gradient
                    conv_result = correlate(output_gradient[i, c], self.kernels[c, k], mode='full')
                    conv_result_resized = np.resize(conv_result, self.input[i, k].shape) # would love to eliminate the need for this
                    input_grad[i, k] += conv_result_resized

        # update parameters
        self.kernels = self.kernels_opt.update(self.kernels, kernels_grad)
        self.b = self.b_opt.update(self.b, bias_grad)

        return input_grad

################# PREPARE DATA 

def preprocess_data(x, y):
    x = x[:, np.newaxis, :, :]        # add channel dimension
    x = x.astype('float32') / 255     # normalization: data ranges between 0 and 1
    y = to_categorical1D(y, 10)       # one-hot encoding
    return shuffler(x, y)             # return shuffled data

with np.load('Examples/data/mnist.npz', allow_pickle=True) as f:
    (train_data), (test_data) = (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])

X_train, y_train = preprocess_data(train_data[0], train_data[1])
X_test, y_test = preprocess_data(test_data[0], test_data[1]) 

X_train, X_val, y_train, y_val = X_train[:5000], X_train[5000:6000], y_train[:5000], y_train[5000:6000]
X_test, y_test = X_test[:1000], y_test[:1000]

print(f"Train Shape : {X_train.shape}")
print(f"Val Shape   : {X_val.shape}")
print(f"Test Shape  : {X_test.shape}", end='\n\n')

################# SET HYPERPARAMETERS

n_samples, n_channels, height, width = X_train.shape
learning_rate= 0.01
epochs=10
batch_size=128

################# DEFINE THE MODEL

# very small model considering the unoptimized CNN and runtime being CPU
clf = NeuralNetwork(Adam(), CCE(), [
    Conv2Dexperimental(in_chnls=n_channels, out_chnls=3, kernel_size=3, stride=2),  # (1, 28, 28) -> (3, 13, 13) 
    ReLu(),
    Conv2Dexperimental(in_chnls=3, out_chnls=6, kernel_size=3, stride=2),           # (3, 13, 13) -> (6, 6, 6)
    ReLu(),
    Flatten(),                                                              # (6, 6, 6)  -> (6*6*6)
    Dense(6*6*6, 10),
    Softmax()
])

################# DEFINE DATA LOADERS AND TRAIN THE MODEL

train_loader = DataLoader(X_train, y_train, batch_size, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size, shuffle=True)
test_loader = DataLoader(X_test, y_test, batch_size, shuffle=True)

history = clf.fit(train_loader, val_loader, epochs, accuracy=True)

################# SAVE THE MODEL
save_object(clf, 'Examples/models/mnist_clf_cnn_experimental.pkl')

################# LOAD THE MODEL
clf = load_object('Examples/models/mnist_clf_cnn_experimental.pkl')

################# EVALUATE HOWEVER YOU LIKE
def plot_loss():    
    loss:dict = history[0]
    history_df = pd.DataFrame({'train': loss['train'], 'val': loss['val']})
    history_df.plot(title='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

def calculate_test_accuracy():
    acc = []
    for (X, y) in test_loader():
        y_pred = clf.predict(X)
        acc.append(clf.acc(y_pred, y))
    return np.mean(acc)

def plot_images_with_predictions():
    fig = plt.figure(figsize=(8, 5)) 
    start = 250 # see any 50 digits starting from this index
    for i in range(start, start+50): 
        plt.subplot(5, 10, (i-start)+1)  
        plt.xticks([]); plt.yticks([])
    
        img = X_test[i] # shape: (1, 28, 28)
        plt.imshow(img[0], cmap=plt.cm.binary) # type: ignore
        
        predicted_label = np.argmax(clf.predict(X_test[i, np.newaxis, :, :]))
        true_label = np.argmax(y_test[i])
        
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)  
    plt.show()

print(f'Test Accuracy: {calculate_test_accuracy():.4f}%')
plot_loss()
plot_images_with_predictions()

################# Accuracy with current hyperparameters: ~85% to ~92% (very unstable)