import numpy as np
np.set_printoptions(precision=3)

import logger

class Layer:
    def __init__(self, dimen, name=None, weights=None, biases=None, verbose=1):

        self.dimen = dimen
        self.__name = name
        self.__weights = weights
        self.__biases = biases
        self.verbose = verbose


    def compile(self, input_dimen):

        # init weights & biases
        if self.__weights is None:
            self.__weights = np.random.rand(input_dimen, self.dimen)
        if self.__biases is None:
            self.__biases = np.random.rand(self.dimen)


    @property
    def name(self):
        return self.__name

    @property
    def weights(self):
        return self.__weights

    @property
    def biases(self):
        return self.__biases

    def get_dimen(self):
        return self.dimen

    def print_info(self):
        logger.v_detail(self.verbose, 'Layer : ', self.name)
        logger.v_detail(self.verbose, '\n-- weights:', self)
        logger.v_detail(self.verbose, self.weights)
        logger.v_detail(self.verbose, '\n-- biases: ')
        logger.v_detail(self.verbose, self.biases)

    def forward(self, input):
        # no impl
        return np.empty

class Dense(Layer):

    def forward(self, input):

        self.print_info()
        forward_result = input.dot(self.weights) + self.biases
        logger.v_detail(self.verbose, '\n-- Layer Result: ', self.name)
        logger.v_detail(self.verbose, forward_result)

        return forward_result