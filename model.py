import numpy as np

import bare_ml
from layer import Layer
from loss import MeanSquaredError
import logger

LAYER_ZERO = 0


class Model():
    def __init__(self, name=None, verbose=1):
        self.name = name
        self.verbose = verbose
        self.layers = []

        self.init_from_env()

    def init_from_env(self):

        print_precision = bare_ml.get_env().get('print_precision', None)
        if print_precision:
            np.set_printoptions(precision=print_precision)

        manual_seed = bare_ml.get_env().get('manual_seed', None)
        if manual_seed:
            np.random.seed(manual_seed)

    @property
    def info(self):
        info = {}
        info['name'] = self.name
        info['env'] = bare_ml.get_env()
        return info

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

        logger.silent(self.verbose, self.info)

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
