from layer import Layer

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
        layers = self.__layers
        dimen = input_dimen
        for layer in layers:
            layer.compile(dimen)
            dimen = layer.get_dimen()

    def fit(self, x=None, y=None, epochs=1):

        layer_input = x
        layers = self.__layers
        for epoch in range(epochs):
            print("\n\n----- ## ----- //// EPOCH: ", str(epoch + 1))
            for idx, layer in enumerate(layers):
                if idx == 0:
                    layer_input = x
                result = layer.forward(layer_input)
                layer_input = result