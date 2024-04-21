import numpy as np

class Dense:
    def __init__(self, neurons_number, inputs_number, learning_rate=0.01) -> None:
        self.weights = np.random.rand(neurons_number, inputs_number) * 0.1
        self.biases = np.zeros((1, neurons_number))
        self.learning_rate = learning_rate
        self.input_values = None

    def forward(self, input_values):
        self.input_values = input_values
        return np.dot(input_values, self.weights.T) + self.biases
    
    def backward(self, derivative):
        self.weights -= self.learning_rate * np.dot(derivative.T, self.input_values)
        self.biases -= self.learning_rate * np.dot(np.ones((1, len(derivative))), derivative)

        return np.dot(derivative, self.weights)

