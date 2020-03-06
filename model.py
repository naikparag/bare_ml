from layer import Layer
from loss import MeanSquaredError
import numpy as np

import logger

LAYER_ZERO = 0

class Model():
    def __init__(self, name=None, verbose=1):
        self.name = name
        self.__layers = []
        self.verbose = verbose

    @property
    def layers(self):
        return self.__layers

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                'The added layer must be an instance of class Layer. Found: ' + str(layer))

        self.__layers.append(layer)

    def compile(self, input_dimen, learning_rate=0.001):
        logger.v_detail(self.verbose, 'MODEL: Compile')
        layers = self.__layers
        dimen = input_dimen
        for layer in layers:
            logger.v_detail(self.verbose, 'MODEL: compiling layer - ', layer.name)
            layer.compile(dimen)
            dimen = layer.get_dimen()

    def fit(self, x, y, epochs=1, verbose=0):
        logger.v_silent(verbose, 'MODEL: Fit')
        layer_input = x
        layers = self.__layers
        result = np.empty
        for epoch in range(epochs):
            logger.v_silent(verbose, '\n----------------------')
            logger.v_silent(verbose, '// EPOCH: ', str(epoch + 1))
            logger.v_silent(verbose, '----------------------')

            for idx, layer in enumerate(layers):
                if idx == LAYER_ZERO:
                    layer_input = x
                result = layer.forward(layer_input)
                layer_input = result

            loss = self.calc_loss(y, result)
            logger.v_silent(verbose, '\n -- LOSS: ', loss)


    def calc_loss(self, y, y_pred):
        return MeanSquaredError.calc(y, y_pred)
