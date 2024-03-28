import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.model_selection import train_test_split

from nn.network import NeuralNetwork
from nn.optim import SGD
from nn.losses import CrossEntropy
from nn.layers import Dense
from nn.activations import ReLu, Softmax
from nn.regularization import Regularization
from nn.utils import to_categorical1D, DataLoader

################# PREPARE DATA 

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], 28*28)  # flattening 28x28 to a 784 vector
    x = x.astype("float32") / 255     # normalization: data ranges between 0 and 1
    y = to_categorical1D(y, 10)       # one-hot encoding
    return x, y

(train_data), (test_data) = mnist.load_data() 

X_train, y_train = preprocess_data(train_data[0], train_data[1])
X_test, y_test = preprocess_data(test_data[0], test_data[1]) 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

################# SET HYPERPARAMETERS

n_samples, n_features = X_train.shape
learning_rate: float = 0.01
momentum:float = 0.9
batch_size:int = 128
rglr: Regularization = Regularization('L2', 0.001)


################# DEFINE THE MODEL

clf = NeuralNetwork(SGD(momentum=momentum), CrossEntropy(), [
    Dense(n_features, 128, rglr), 
    ReLu(),  
    Dense(128, 64, rglr),  
    ReLu(),
    Dense(64, 32, rglr),
    ReLu(),
    Dense(32, 10),
    Softmax()
])

################# DEFINE DATA LOADERS AND TRAIN THE MODEL

train_loader = DataLoader(X_train, y_train, batch_size, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size, shuffle=True)

history = clf.fit(train_loader, val_loader, epochs=10, verbose=True)


################# EVALUATE HOWEVER YOU LIKE

def plot_loss():    
    train_loss, val_loss = history
    print(len(train_loss), len(val_loss))
    history_df = pd.DataFrame({'train': train_loss, 'val': val_loss})
    history_df.plot(title="Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.show()
plot_loss()

def calculate_accuracy():
    test_loader = DataLoader(X_test, y_test, batch_size, shuffle=True)
    correct = 0
    for (X, y) in test_loader():
        y_pred = clf.predict(X)
        correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

    return (correct / len(y_test)) * 100

print(f"Accuracy: {calculate_accuracy()}%")


# def plot_images_with_predictions():
#     fig = plt.figure(figsize=(8, 5))  # figure size in inches
    
#     for i in range(50): 
#         plt.subplot(5, 10, i+1)  
#         plt.xticks([]); plt.yticks([])  # remove markings
        
#         img = x_test[i].reshape((28, 28))
#         plt.imshow(img, cmap=plt.cm.binary)
        
#         predicted_label = np.argmax(network.predict(x_test[i]))
#         true_label = np.argmax(y_test[i])
        
#         color = 'green' if predicted_label == true_label else 'red'
#         plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)  
#     plt.show()
# plot_images_with_predictions()

# Accuracy with current hyperparameters: ~94%

