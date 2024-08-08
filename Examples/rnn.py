import numpy as np

from nn.activations import Tanh
from nn.layers import RNN
from nn.losses import MSE
from nn.optim import Adam
from nn.regularization import Regularization

X = np.random.randn(1000, 10, 5)
Y = np.random.randn(1000, 1)

model = RNN(5, 10, 1, Tanh(), Regularization('L2', 0.01))

criterion = MSE()

model.initialize(Adam(0.001))

# Training loop: RNN isn't compatible with `nn.network.NeuralNetwork.fit()` method
for epoch in range(10):
    hidden = np.zeros((1000, 10)) # start blank hidden state

    output, hidden = model.forward(X, hidden)
    loss = criterion(Y, output[:, -1, :])
    loss_grad = criterion.grad(Y, output[:, -1, :]).clip(-1, 1)
    
    # grad needs to retain the same shape as output
    out_grad = np.zeros_like(output)
    out_grad[:, -1, :] = loss_grad
    
    model.backward(out_grad)
    
    print(f'Epoch {epoch}: Loss {np.mean(loss)}')