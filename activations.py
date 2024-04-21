import numpy as np

class Relu:
    def __init__(self) -> None:
        self.input_values = None

    def forward(self, input_values):

        self.input_values = input_values
        input_values[input_values < 0] = 0

        return input_values
    
    def backward(self, derivative):
        derivative[self.input_values < 0] = 0

        return derivative
    
    
class Softmax:
    def __init__(self) -> None:
        self.softmax_values = None

    def forward(self, input_values):
        output = np.empty_like(input_values)

        for index, element in enumerate(input_values):
            output[index] = np.array([np.exp(el)/np.sum(np.exp(element)) for el in element])


        self.softmax_values = output

        return output
    
    def backward(self, derivative):
        jacobian_matrix = np.zeros((len(self.softmax_values[0]), len(self.softmax_values[0])))

        output = np.empty_like(derivative)

        for index, softmax_sample in enumerate(self.softmax_values):
            for i in range(len(softmax_sample)):
                for j in range(len(softmax_sample)):
                    if i == j:
                        val = softmax_sample[i] * (1 - softmax_sample[j])
                    else:
                        val = -softmax_sample[i] * softmax_sample[j] 

                    jacobian_matrix[i][j] = val


            output[index] = np.dot(np.expand_dims(derivative[index], axis=0), jacobian_matrix)[0]

        return output
    
