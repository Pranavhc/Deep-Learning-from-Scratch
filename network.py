from typing import List
from layer import Layer
from loss_fn import Loss

class NeuralNetwork:
    def __init__(self, layers: List[Layer]) -> None:
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def train(self, loss: Loss, x_train, y_train, epochs=1000, learning_rate=0.1, verbose=True, verbose_interval=100):
        for e in range(epochs+1):
            error = 0
            accuracy = 0
            for x, y in zip(x_train, y_train):
                # forward
                output = self.predict(x)

                # error
                error += loss(y, output)
                accuracy += int(y.argmax() == output.argmax())
                #backward
                grad = loss.prime(y, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, learning_rate)

            error /= len(x_train)
            
            if verbose and e % verbose_interval == 0:
                print(f"{e}/{epochs} - error = {error} - accuracy = {accuracy/len(y_train)}")
            
