import numpy as np

from layer import Layer
from loss import MeanSquaredError
import logger

LAYER_ZERO = 0


class Model():
    def __init__(self, name=None, verbose=1):
        self.name = name
        self.verbose = verbose
        self.layers = []

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                'Added layer must be an instance of class Layer. Found: ' + str(layer))

        self.layers.append(layer)

    def compile(self, input_dimen, learning_rate=0.001):
        logger.detail(self.verbose, 'MODEL: Compile')
        layers = self.layers
        dimen = input_dimen
        for layer in layers:
            logger.detail(
                self.verbose, 'MODEL: compiling layer - ', layer.name)
            layer.compile(dimen)
            dimen = layer.info['dimen']

    def fit(self, x, y, epochs=1, verbose=0):
        logger.silent(verbose, 'MODEL: Fit')
        layer_input = x
        result = np.empty
        for epoch in range(epochs):
            logger.silent(verbose, '\n----------------------')
            logger.silent(verbose, '// EPOCH: ', str(epoch + 1))
            logger.silent(verbose, '----------------------')

            for idx, layer in enumerate(self.layers):
                if idx == LAYER_ZERO:
                    layer_input = x
                result = layer.forward(layer_input)
                layer_input = result

            loss = self.calc_loss(y, result)
            logger.silent(verbose, '\n -- MODEL LOSS: ', loss)

    def calc_loss(self, y, y_pred):
        return MeanSquaredError.calc(y, y_pred)
