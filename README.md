### A from-scratch Implementation of a Neural Network in Python
##### I am building this library to improve my understanding of deep learning techniques.


Usage Example:
```python

# Imports
from nn.network import NeuralNetwork
from nn.optim import Adam
from nn.losses import CategoricalCrossEntropy as CCE
from nn.layers import Dense, Dropout
from nn.regularization import Regularization as Rglr
from nn.activations import ReLu, Softmax
from nn.utils import DataLoader, save_object, load_object

# Define your model
clf = NeuralNetwork(Adam(), CCE(), [
    Dense(784, 256, Rglr('L2', 0.1)), 
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

# Prepare your datasets and define their data loaders
train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=True)
test_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)

# train your model 
history = clf.fit(train_loader, val_loader, epochs=10, verbose=True)

# save and load 
save_object(clf, "Examples/models/clf.pkl")
clf = load_object("Examples/models/clf.pkl")

# predict and test your model
correct = 0
for (X, y) in test_loader():
    y_pred = clf.predict(X)
    correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

print(f"Accuracy: {(correct / len(y_test)) * 100}")

```