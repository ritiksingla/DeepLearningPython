import numpy as np
from numpy import ndarray
import sys

sys.path.append("..")
from utils import assert_same_shape


class Loss(object):
    def __init__(self):
        pass

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        assert_same_shape(prediction, target)
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def _output(self) -> float:
        raise NotImplementedError()

    def backward(self) -> ndarray:
        self.input_grad_ = self._input_grad()
        assert_same_shape(self.prediction, self.input_grad_)
        return self.input_grad_

    def _input_grad(self) -> ndarray:
        raise NotImplementedError()
