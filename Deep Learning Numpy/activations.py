import numpy as np
from numpy import ndarray
from base import Operation
from utils import softmax


class Linear(Operation):
    '''
    Linear Activation Function
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return self.input_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return output_grad


class Sigmoid(Operation):
    '''
    Sigmoid Activation Function
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return 1.0 / (1.0 + np.exp(-1.0 * self.input_))

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return self.output_ * (1 - self.output_) * output_grad


class Tanh(Operation):
    '''
    Tanh Activation Function
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return np.tanh(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return output_grad * (1 - self.output_ * self.output_)


class ReLU(Operation):
    '''
    ReLU Activation Function
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return np.clip(self.input_, 0, None)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return output_grad * (self.output_ >= 0)


class Softmax(Operation):
    '''
    Softmax Activation Function
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return softmax(self.input_)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute input gradient'''
        return output_grad
