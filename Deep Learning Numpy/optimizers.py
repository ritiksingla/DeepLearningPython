import numpy as np
from numpy import ndarray


class Optimizer(object):
    '''
    Base class for neural network optimizer
    '''

    def __init__(self, lr: float = 0.01):
        self.lr = lr
        self.first = True

    def step(self):
        raise NotImplementedError()


class SGD(Optimizer):
    def __init__(self, lr: float, momentum=0, dampening=0, weight_decay=0):
        super().__init__(lr)
        self.momentum = momentum
        self.dampening = dampening
        self.weight_decay = weight_decay

    def step(self):
        if self.first:
            self.velocities = [np.zeros_like(param) for param in self.net.param_grads()]
            self.first = False
        for (param, param_grad, vel) in zip(
            self.net.params(), self.net.param_grads(), self.velocities
        ):
            vel *= self.momentum
            vel += (1 - self.dampening) * param_grad
            param -= self.lr * vel
