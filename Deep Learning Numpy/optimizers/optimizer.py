import numpy as np
from numpy import ndarray
import sys

sys.path.append("..")


class Optimizer(object):
    '''
    Base class for neural network optimizer
    '''

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self):
        raise NotImplementedError()
