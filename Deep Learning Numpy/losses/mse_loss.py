import numpy as np
from numpy import ndarray
from losses.loss import Loss


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
