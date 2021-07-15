import numpy as np
from numpy import ndarray
from optimizers.optimizer import Optimizer


class SGD(Optimizer):
    def __init__(self, lr: float = 0.01):
        super().__init__(lr)

    def step(self):
        for (param, param_grad) in zip(self.net.params(), self.net.param_grads()):
            param -= self.lr * param_grad
