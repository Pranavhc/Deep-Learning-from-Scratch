## Neural Networks built from scratch

All the mathematical operations and the neural network layers are implemented from scratch using numpy. The neural network is built in a modular way, where each layer is a class that can be added to the network. The network class is responsible for the forward and backward pass, and the optimization is done using the optimizer class. The network can be trained using the fit method, and the model can be saved and loaded using the save_object and load_object functions. 

Usage Example:
```python
from nn.network import NeuralNetwork
from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Dense, Dropout
from nn.regularization import Regularization as Rglr
from nn.activations import ReLu, Softmax
from nn.utils import DataLoader, save_object, load_object

layers = [
    Dense(784, 128, Rglr('L2', 0.1)), 
    ReLu(),

    Dense(128, 64),  
    Dropout(0.3),
    ReLu(),

    Dense(64, 32),
    Dropout(0.2),
    ReLu(),

    Dense(32, 10),
    Softmax()
]

clf = NeuralNetwork(Adam(), CCE(), layers)

train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=True)

history = clf.fit(train_loader, val_loader, epochs=10, accuracy=True)
```

I will keep adding more features and improvements as I learn more about Deep Learning.