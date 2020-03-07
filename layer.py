import logger
import numpy as np
np.set_printoptions(precision=3)


class Layer:
    def __init__(self, dimen, name=None, weights=None, biases=None, verbose=1):

        self.dimen = dimen
        self.name = name
        self.weights = weights
        self.biases = biases
        self.verbose = verbose

    @property
    def info(self):
        info = {}
        info['name'] = self.name
        info['dimen'] = self.dimen
        return info

    def compile(self, input_dimen):

        # init weights & biases
        if self.weights is None:
            self.weights = np.random.rand(input_dimen, self.dimen)
        if self.biases is None:
            self.biases = np.random.rand(self.dimen)


    def get_dimen(self):
        return self.dimen

    def print_info(self):
        logger.detail(self.verbose, 'Layer : ', self.name)
        logger.detail(self.verbose, '\n-- weights:', self)
        logger.detail(self.verbose, self.weights)
        logger.detail(self.verbose, '\n-- biases: ')
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
