import numpy as np
from numpy import ndarray


class Optimizer(object):
    '''
    Base class for neural network optimizer
    '''

    def __init__(self, lr: float = 0.01):
        self.lr = lr

    def step(self, local_step: int):
        raise NotImplementedError()


class SGD(Optimizer):
    '''
    Implements Stochastic Gradient Descent Optimization with optional momentum.
    Momentum helps in generalizing the Gradient Descent step based on history of batches.
    Batches may differ largely in distribution and hence almost always optimal over SGD without momentum.
    Momentum help in fast convergence and ending up in tighter region around minima.

    Parameters
    ----------
    lr : float
        learning rate

    momentum : float in range [0, 1]
        acceleration for momentum

    dampening : float in range [0, 1]
        friction for momentum

    nesterov : bool
        nesterov version

    '''

    def __init__(
        self, lr: float, momentum: float = 0.0, dampening: float = 0.0, nesterov=False
    ):
        super().__init__(lr)
        self.momentum = momentum
        self.dampening = dampening
        self.nesterov = nesterov

    def step(self, local_step: int):
        '''
        Take a single step toward optimal parameters to minimize the given loss

        Parameters
        ----------
        local_step : int
            Current batch number
        '''
        if local_step == 1:
            self.v_t = [np.zeros_like(param) for param in self.net.param_grads()]
        for (param, grad, v_t) in zip(
            self.net.params(), self.net.param_grads(), self.v_t
        ):
            v_t *= self.momentum
            v_t += (1 - self.dampening) * grad
            if self.nesterov:
                param -= self.lr * (grad + v_t * self.momentum)
            else:
                param -= self.lr * v_t


class RMSProp(Optimizer):
    '''
    Implements RMSProp Optimization.
    Useful for smoothening out oscillations in preforming Gradient Descent.
    Penalize large gradients more as compared to small gradients.
    '''

    def __init__(
        self, lr: float = 1e-3, momentum: float = 0.0, rho: float = 0.9, epsilon=1e-07
    ):
        super().__init__(lr)
        self.momentum = momentum
        self.rho = rho
        self.epsilon = epsilon

    def step(self, local_step: int):
        if local_step == 1:
            self.m_t = [np.zeros_like(param) for param in self.net.param_grads()]

        for (param, grad, m_t) in zip(
            self.net.params(), self.net.param_grads(), self.m_t
        ):
            m_t *= self.rho  # mean square history
            m_t += (1 - self.rho) * grad * grad
            mom = self.momentum * m_t + self.lr * grad / (np.sqrt(m_t) + self.epsilon)
            param -= mom


class Adam(Optimizer):
    '''
    Implements Adam Optimization (RMSProp + SGD with momentum) Adaptive Moment Estimation
    While momentum accelerates our search in direction of minima,
    RMSProp impedes our search in direction of oscillations.
    Therefore Adam is almost always best to use for complex networks.
    '''

    def __init__(
        self,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon=1e-07,
        nesterov: bool = False,
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.nesterov = nesterov

    def step(self, local_step: int):
        if local_step == 1:
            self.v_t = [np.zeros_like(param) for param in self.net.param_grads()]
            self.m_t = [np.zeros_like(param) for param in self.net.param_grads()]
        beta1_power = np.power(self.beta1, local_step)
        beta2_power = np.power(self.beta2, local_step)
        alpha = self.lr * np.sqrt(1 - beta2_power) / (1 - beta1_power)
        for (param, grad, m_t, v_t) in zip(
            self.net.params(), self.net.param_grads(), self.m_t, self.v_t
        ):
            v_t *= self.beta1
            v_t += (1 - self.beta1) * grad

            m_t *= self.beta2
            m_t += (1 - self.beta2) * grad * grad

            if self.nesterov:
                param -= (
                    alpha
                    * (v_t * self.beta1 + (1 - self.beta1) * grad)
                    / (np.sqrt(m_t) + self.epsilon)
                )
            else:
                param -= alpha * v_t / (np.sqrt(m_t) + self.epsilon)
