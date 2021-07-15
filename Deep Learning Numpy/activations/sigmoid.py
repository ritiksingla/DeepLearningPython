import numpy as np
from numpy import ndarray

import sys

sys.path.append("..")
from base import Operation


class Sigmoid(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return self.output_ * (1 - self.output_) * output_grad
