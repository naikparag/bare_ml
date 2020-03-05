import numpy as np
np.set_printoptions(precision=3)

class Layer:
    def __init__(self, dimen, **kwargs):

        allowed_kwargs = {'name',
                          'weights',
                          'biases'
                          }
        for kwarg in kwargs:
            if kwarg not in allowed_kwargs:
                raise TypeError('Keyword argument not understood:', kwarg)

        self.dimen = dimen
        self.__name = kwargs['name']

        
    def compile(self, input_dimen):

        # init weights & biases
        self.__weights = np.random.rand(input_dimen, self.dimen)
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
        print("Layer : ", self.name)
        # print("\n -- input:")
        # print(input)
        print("\n-- weights:")
        print(self.weights)
        print("\n-- biases: ")
        print(self.biases)

    def forward(self, input):
        # no impl
        return np.empty

class Dense(Layer):

    def forward(self, input):

        self.print_info()
        forward_result = input.dot(self.weights) + self.biases
        print("\n-- RESULTS: ")
        print(forward_result)

        return forward_result
