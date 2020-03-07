import numpy as np

import bare_ml
import logger


class Layer:
    def __init__(self, dimen, name=None, weights=None, biases=None, verbose=1):

        self.dimen = dimen
        self.name = name
        self.weights = weights
        self.biases = biases
        self.verbose = verbose

        self.init_from_env()

    @property
    def info(self):
        info = {}
        info['name'] = self.name
        info['dimen'] = self.dimen
        return info

    def init_from_env(self):

        manual_seed = bare_ml.get_env().get('manual_seed', None)
        if manual_seed:
            np.random.seed(manual_seed)

    def compile(self, input_dimen):

        # init weights & biases
        if self.weights is None:
            self.weights = np.random.randint(
                0, 10, size=(input_dimen, self.dimen))*0.1

        if self.biases is None:
            self.biases = np.random.randint(0, 10, self.dimen)*0.1

    def get_dimen(self):
        return self.dimen

    def print_info(self):
        logger.detail(self.verbose, 'Layer : ', self.name)
        logger.detail(self.verbose, '\n-- weights:', self.name)
        logger.detail(self.verbose, self.weights)
        logger.detail(self.verbose, '\n-- biases:', self.name)
        logger.detail(self.verbose, self.biases)

    def forward(self, input):
        # no impl
        return np.empty


class Dense(Layer):

    def forward(self, input):

        self.print_info()
        forward_result = input.dot(self.weights) + self.biases
        logger.detail(self.verbose, '\n-- Layer Result: ', self.name)
        logger.detail(self.verbose, forward_result)

        return forward_result
