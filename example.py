import numpy as np
import random

import layer
import model

# ----- example 1
inputs = np.array([[0, 1]])
y = np.array([1])

weights = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2]])
biases = np.array([[0.1, 0.1, 0.1, 0.1]])
weights2 = np.array([[0.1, 0.1, 0.1, 0.1], [0.2, 0.2, 0.2, 0.2], [
                    0.4, 0.4, 0.4, 0.4], [0.5, 0.5, 0.5, 0.5]])
weights3 = np.array([[0.3], [0.3], [0.3], [0.3]])
biases3 = np.array([[0.1]])

# dense = layer.Dense(4, name="dense", weights=weights, biases=biases)
# dense2 = layer.Dense(4, name="dense2", weights=weights2, biases=biases)
# dense3 = layer.Dense(1, name="dense3", weights=weights3, biases=biases3)

# model = model.Model(name="my model")
# model.add(dense)
# model.add(dense2)
# model.add(dense3)
# model.compile(2)
# model.fit(x=inputs, y=y, epochs=2)

# ---- example 2 XOR


def gen_xor(count):
    xor_seed = [[0, 0], [0, 1], [1, 0], [1, 1]]
    x = random.choices(xor_seed, k=count)
    def xor(x): return 1 if x[0] == x[1] else 0
    y = np.apply_along_axis(xor, 1, x)
    return np.array(x), np.array(y)


x, y = gen_xor(100)
xor_weights = np.array([[0.1, 0.1, 0.1, 0.1], [0.1, 0.1, 0.1, 0.1]])
xor_weights2 = np.array([[0.2], [0.2], [0.2], [0.2]])
dense1 = layer.Dense(4, name="dense1", weights=np.full((2, 4), 0.1), verbose=1)
dense2 = layer.Dense(4, name="dense2", weights=np.full((4, 4), 0.1), verbose=1)
dense3 = layer.Dense(4, name="dense3", weights=np.full((4, 4), 0.1), verbose=1)
dense4 = layer.Dense(1, name="dense4", weights=np.full((4, 1), 0.1), verbose=1)

model = model.Model(name="xor_model")
model.add(dense1)
model.add(dense2)
model.add(dense3)
model.add(dense4)
model.compile(2)
model.fit(x=x, y=y, epochs=1)
