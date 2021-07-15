import numpy as np
from numpy import ndarray

import sys

sys.path.append("..")
from base import Operation


class Linear(Operation):
    def __init__(self):
        super().__init__()

    def _output(self):
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return output_grad
