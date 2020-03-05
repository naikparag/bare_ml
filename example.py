import layer
import model
import numpy as np


inputs = np.array([[0, 1]])

weights = np.array([0.01, 0.02])
biases = np.array([0.1, 0.1])

dense = layer.Dense(4, name="dense", weights=weights, biases=biases)
dense2 = layer.Dense(4, name="dense2", weights=weights, biases=biases)

model = model.Model(name="my model")
model.add(dense)
model.add(dense2)
model.compile(2)
model.fit(x=inputs, epochs=2)