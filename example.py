import layer
import model
import numpy as np


inputs = np.array([[0, 1]])
y = np.array([1])

weights = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
biases = np.array([[0.1, 0.1, 0.1, 0.1]])
weights2 = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]])
weights3 = np.array([[0.3], [0.3], [0.3], [0.3]])
biases3 = np.array([[0.1]])

dense = layer.Dense(4, name="dense", weights=weights, biases=biases)
dense2 = layer.Dense(4, name="dense2", weights=weights2, biases=biases)
dense3 = layer.Dense(1, name="dense3", weights=weights3, biases=biases3)

model = model.Model(name="my model")
model.add(dense)
model.add(dense2)
model.add(dense3)
model.compile(2)
model.fit(x=inputs, y=y, epochs=2)