### A from-scratch Implementation of a Neural Network in Python
##### I've wrote this for learning purposes.

Usage Example:
```python
from nn import *

rglr = Regularization('L2', 0.001)

# Define your model
clf = NeuralNetwork(SGD(momentum=0.9), CrossEntropy(), [
    Dense(28*28, 64, rglr), 
    ReLu(),
    Dense(64, 32, rglr),
    ReLu(),
    Dense(32, 10),
    Softmax()
])

# Prepare you data and define data loaders
train_loader = DataLoader(X_train, y_train, batch_size=128, shuffle=True)
val_loader = DataLoader(X_val, y_val, batch_size=128, shuffle=True)

test_loader = DataLoader(X_test, y_test, batch_size=128, shuffle=False)

# train the model
history = clf.fit(train_loader, val_loader, epochs=10)

# save or load
save_object(clf, "Examples/models/clf.pkl")
clf = load_object("Examples/models/clf.pkl")

# predict and test the model
correct = 0
for (X, y) in test_loader():
    y_pred = clf.predict(X)
    correct += np.sum(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1))

print(f"Accuracy: {(correct / len(y_test)) * 100}")

```