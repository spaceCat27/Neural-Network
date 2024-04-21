import numpy as np

class Categorical_crossentropy:
    def __init__(self) -> None:
        pass

    def accuracy(self, input_values, correct_values):
        one_hot = self.change_vector(correct_values=correct_values)
        
        correct_guesses = 0
        total_guesses = len(one_hot)

        for input_ar, correct_value in zip(input_values, one_hot):
            if np.argmax(input_ar) == np.argmax(correct_value):
                correct_guesses += 1

        return correct_guesses/total_guesses

    def loss(self, input_values, correct_values):
        one_hot = self.change_vector(correct_values=correct_values)

        
        input_values_cliped = np.clip(input_values, 0.1e-20, np.max(input_values))

        output = np.empty([len(input_values_cliped), 1])

        for index, (input_ar, one_hot) in enumerate(zip(input_values_cliped, one_hot)):
            output[index][0] = -np.log(np.dot(input_ar, one_hot))

        return output

    def backward(self, input_values, correct_values):
        one_hot = self.change_vector(correct_values=correct_values)
        
        return -one_hot/input_values

    def change_vector(self, correct_values):
        one_hot = correct_values
        if len(correct_values.shape) == 1:
            one_hot = np.zeros((len(correct_values), 3))
            for index, val in enumerate(correct_values):
                one_hot[index][val] = 1
        
        return one_hot


