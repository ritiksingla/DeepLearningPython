import numpy as np
from numpy import ndarray
from utils import assert_same_shape
from scipy.special import xlogy


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


class MSELoss(Loss):
    def __init__(self) -> None:
        # pass
        super().__init__()

    def _output(self) -> float:
        loss = np.linalg.norm(self.prediction - self.target) ** 2
        loss /= self.prediction.shape[0]
        return loss

    def _input_grad(self) -> ndarray:
        return 2.0 * (self.prediction - self.target) / self.prediction.shape[0]


class L1Loss(Loss):
    def __init__(self) -> None:
        # pass
        super().__init__()

    def _output(self) -> float:
        loss = np.sum(np.fabs(self.prediction - self.target))
        loss /= self.prediction.shape[0]
        return loss

    def _input_grad(self) -> ndarray:
        grad = np.sign(self.prediction - self.target)
        return grad / self.prediction.shape[0]


class BCELoss(Loss):
    def __init__(self) -> None:
        # pass
        super().__init__()

    def forward(self, prediction: ndarray, target: ndarray) -> float:
        self.prediction = prediction
        self.target = target
        loss_value = self._output()
        return loss_value

    def _output(self) -> float:
        eps = np.finfo(self.prediction.dtype).eps
        self.prediction = np.clip(self.prediction, eps, 1 - eps)
        if self.prediction.shape[1] == 1:
            self.prediction = np.append(1 - self.prediction, self.prediction, axis=1)

        if self.target.shape[1] == 1:
            self.target = np.append(1 - self.target, self.target, axis=1)
        return -xlogy(self.target, self.prediction).sum() / self.prediction.shape[0]

    def _input_grad(self) -> ndarray:
        grad = self.prediction - self.target
        return grad / self.prediction.shape[0]
