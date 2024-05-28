import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from nn.network import NeuralNetwork
from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Dense, Dropout
from nn.activations import ReLu, Softmax
from nn.utils import DataLoader, to_categorical1D, shuffler, save_object, load_object

################# PREPARE DATA 

def preprocess_data(x, y):
    x = x.reshape(x.shape[0], 28*28)  # flattening 28x28 matrix to a 784 vector
    x = x.astype('float32') / 255     # normalization: data ranges between 0 and 1
    y = to_categorical1D(y, 10)       # one-hot encoding
    return shuffler(x, y)             # return shuffled data

with np.load('Examples/data/mnist.npz', allow_pickle=True) as f:
    (train_data), (test_data) = (f['x_train'], f['y_train']), (f['x_test'], f['y_test'])

X_train, y_train = preprocess_data(train_data[0], train_data[1])
X_test, y_test = preprocess_data(test_data[0], test_data[1]) 

X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

################# SET HYPERPARAMETERS

n_samples, n_features = X_train.shape
learning_rate= 0.01
epochs=10
batch_size=256

################# DEFINE THE MODEL

clf = NeuralNetwork(Adam(), CCE(), [
    Dense(n_features, 256), 
    Dropout(0.3),
    ReLu(),  

    Dense(256, 128), 
    Dropout(0.3),
    ReLu(),

    Dense(128, 64),  
    Dropout(0.3),
    ReLu(),
    
    Dense(64, 32),
    Dropout(0.2),
    ReLu(),
    
    Dense(32, 10),
    Softmax()
])

################# DEFINE DATA LOADERS AND TRAIN THE MODEL

train_loader = DataLoader(X_train, y_train, batch_size, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size, shuffle=True)
test_loader = DataLoader(X_test, y_test, batch_size, shuffle=True)

history = clf.fit(train_loader, val_loader, epochs, accuracy=True)

################# SAVE THE MODEL
save_object(clf, 'Examples/models/mnist_clf.pkl')

################# LOAD THE MODEL
clf = load_object('Examples/models/mnist_clf.pkl')

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
        
        img = X_test[i].reshape((28, 28))
        plt.imshow(img, cmap=plt.cm.binary) # type: ignore
        
        predicted_label = np.argmax(clf.predict(X_test[i]))
        true_label = np.argmax(y_test[i])
        
        color = 'green' if predicted_label == true_label else 'red'
        plt.xlabel("{} ({})".format(predicted_label, true_label), color=color)  
    plt.show()

print(f'Test Accuracy: {calculate_test_accuracy():.4f}%')
plot_loss()
plot_images_with_predictions()

################# Accuracy with current hyperparameters: ~97%