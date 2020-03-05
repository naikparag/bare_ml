import numpy as np


class MeanSquaredError():

    @staticmethod
    def calc(y, y_pred):
        return np.square(y_pred - y).sum()
