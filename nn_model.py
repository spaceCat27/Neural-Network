import numpy as np


class Model:
    def __init__(self, loss) -> None:
        self.layers = []
        self.activations = []
        self.loss = loss

    def train(self, epochs, X, y, verbose=False):
        for epoch in range(epochs):
            output = X
            for layer, acitvaion in zip(self.layers, self.activations):
                output = layer.forward(output)
                output = acitvaion.forward(output)

            acc = self.loss.accuracy(output, y)
            l = self.loss.loss(output, y)

            if epoch % 100 == 0 and verbose:
                print(f'e:{epoch} loss: {np.mean(l, axis=0)} accuracy: {acc}')

            output = self.loss.backward(output, y)

            for layer, acitvaion in zip(reversed(self.layers), reversed(self.activations)):
                output = acitvaion.backward(output)
                output = layer.backward(output)


    def predict(self, input_values):
        output = input_values
        for layer, acitvaion in zip(self.layers, self.activations):
            output = layer.forward(output)
            output = acitvaion.forward(output)

        return output
    

    def evaluate(self, X, y):
        output = X
        for layer, acitvaion in zip(self.layers, self.activations):
            output = layer.forward(output)
            output = acitvaion.forward(output)

        acc = self.loss.accuracy(output, y)
        l = self.loss.loss(output, y)

        
        print(f'loss: {np.mean(l, axis=0)} accuracy: {acc}')      


    def add(self, layer, activation):
        self.layers.append(layer)
        self.activations.append(activation)