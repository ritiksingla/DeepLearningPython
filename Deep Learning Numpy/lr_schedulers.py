import numpy as np
from optimizers import Optimizer


class _LRScheduler(object):
    def __init__(self, optimizer: Optimizer):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer

    def forward(self, epoch):
        self.optimizer.lr = self.get_lr(self.optimizer.lr, epoch)

    def get_lr(self, old_lr, epoch):
        raise NotImplementedError()


class ReduceLROnPlateau(object):
    '''
    Reduce Learning Rate on Plateau
    Parameters
    ----------
    optimizer : Optimizer
        optimizer used for gradient descent
    mode : str either 'min' or 'max'
        whether metrics to forward function decrease or increase
        for example (error or accuracy)
    '''

    def __init__(
        self,
        optimizer: Optimizer,
        mode: str = 'max',
        factor: float = 0.1,
        patience: int = 10,
        threshold: float = 1e-4,
        min_lr: float = 0.0,
        eps: float = 1e-8,
    ):
        if not isinstance(optimizer, Optimizer):
            raise TypeError('{} is not an Optimizer'.format(type(optimizer).__name__))
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.threshold = threshold
        self.eps = eps
        self.best = None
        self.num_bad_epochs = 0

    def forward(self, metrics, epoch: int):
        if self.is_better(metrics, self.best):
            self.best = metrics
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self._reduce_lr(epoch)
            self.num_bad_epochs = 0

    def is_better(self, a, best):
        if best is None:
            return True
        if self.mode == 'min':
            rel_epsilon = 1.0 - self.threshold
            return a < best * rel_epsilon
        elif self.mode == 'max':
            rel_epsilon = 1.0 + self.threshold
            return a > best * rel_epsilon

    def _reduce_lr(self, epoch):
        old_lr = self.optimizer.lr
        new_lr = max(old_lr * self.factor, self.min_lr)
        if old_lr - new_lr > self.eps:
            self.optimizer.lr = new_lr
            print(
                'Epoch {} : reducing learning rate from {:5f} to {:5f}'.format(
                    epoch, old_lr, new_lr
                )
            )


class ExponentialLR(_LRScheduler):
    def __init__(self, optimizer, gamma):
        super().__init__(optimizer)
        self.gamma = gamma

    def get_lr(self, old_lr, epoch):
        new_lr = old_lr * np.exp(-self.gamma)
        print(
            'Epoch {} : reducing learning rate from {:5f} to {:5f}'.format(
                epoch, old_lr, new_lr
            )
        )
        return new_lr
