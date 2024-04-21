import numpy as np
from layers import Dense
from activations import Relu, Softmax
from costs import Categorical_crossentropy
from sklearn.model_selection import train_test_split
from sklearn import datasets
from nn_model import Model

iris_data = datasets.load_iris()
X = iris_data['data']
y = iris_data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

lr = 0.001
dense1 = Dense(neurons_number=24, inputs_number=4, learning_rate=lr)
relu1 = Relu()
dense2 = Dense(neurons_number=3, inputs_number=24, learning_rate=lr)
softmax2 = Softmax()
loss = Categorical_crossentropy()

epochs = 1_001

normalized_x = (X_train - np.min(X_train)) / (np.max(X_train) - np.min(X_train))
normalized_x2 = (X_test - np.min(X_test)) / (np.max(X_test) - np.min(X_test))


model = Model(loss=loss)
model.add(dense1, relu1)
model.add(dense2, softmax2)

model.train(epochs=epochs, X=normalized_x, y=y_train, verbose=True)
model.evaluate(normalized_x2, y=y_test)


print(model.predict(normalized_x2))
