from layer import Layer
from loss import MeanSquaredError
import numpy as np

import logger

LAYER_ZERO = 0

class Model():
    def __init__(self, name=None):
        self.name = name
        self.__layers = []

    @property
    def layers(self):
        return self.__layers

    def add(self, layer):
        if not isinstance(layer, Layer):
            raise TypeError(
                'The added layer must be an instance of class Layer. Found: ' + str(layer))

        self.__layers.append(layer)


    def compile(self, input_dimen, learning_rate=0.001):
        logger.debug('MODEL: Compile')
        layers = self.__layers
        dimen = input_dimen
        for layer in layers:
            logger.debug('MODEL: compiling layer - ', layer.name)
            layer.compile(dimen)
            dimen = layer.get_dimen()

    def fit(self, x, y, epochs=1, verbose=0):
        logger.log(verbose, logger.V_DETAIL,'MODEL: Fit')
        layer_input = x
        layers = self.__layers
        result = np.empty
        for epoch in range(epochs):
            logger.log(verbose, logger.V_SILENT, '\n----------------------')
            logger.log(verbose, logger.V_SILENT, '// EPOCH: ', str(epoch + 1))
            logger.log(verbose, logger.V_SILENT, '----------------------')

            for idx, layer in enumerate(layers):
                if idx == LAYER_ZERO:
                    layer_input = x
                result = layer.forward(layer_input)
                layer_input = result

            loss = self.calc_loss(y, result)
            logger.log(verbose, logger.V_SILENT, '\n -- LOSS: ', loss)


    def calc_loss(self, y, y_pred):
        return MeanSquaredError.calc(y, y_pred)
