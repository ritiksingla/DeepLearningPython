import numpy as np
from numpy import ndarray
from utils import assert_same_shape

# Inherited by Activation Functions and ParamOperation
class Operation(object):
    '''
    Base class for operation in a Neural Network
    '''

    def __init__(self):
        pass

    def forward(self, input_: ndarray):
        '''
        Stores input in the self._input instance variable
        Calls the self._output() function.
        '''
        self.input_ = input_
        self.output_ = self._output()
        return self.output_

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Calls the self._input_grad() function
        '''
        assert_same_shape(output_grad, self.output_)
        self.input_grad_ = self._input_grad(output_grad)
        assert_same_shape(self.input_grad_, self.input_)
        return self.input_grad_

    def _output(self) -> ndarray:
        '''
        The _output method must be defined for each Operation
        '''
        raise NotImplementedError()

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        The _input_grad method must be defined for each Operation
        Compute dL/Z_(i - 1) from dL/dZ_i
        '''
        raise NotImplementedError()


# Inherited by layers containing params
class ParamOperation(Operation):
    def __init__(self, param: ndarray):
        super().__init__()
        self.param_ = param

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output_, output_grad)

        self.input_grad_ = self._input_grad(output_grad)
        self.param_grad_ = self._param_grad(output_grad)

        assert_same_shape(self.input_, self.input_grad_)
        assert_same_shape(self.param_, self.param_grad_)
        return self.input_grad_

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Every subclass of ParamOperation must implement _param_grad.
        Compute dL/dw_i, dL/db_i from dL/dZ_i
        '''
        raise NotImplementedError()


class WeightMultiply(ParamOperation):
    '''
    Weight multiplication operation for neural network
    '''

    def __init__(self, W: ndarray):
        super().__init__(W)

    def _output(self) -> ndarray:
        '''Compute Output'''
        # (None, l) * (r, l).T
        return np.dot(self.input_, self.param_.T)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute Input Gradients dL/dz_i = dL/dz_(i + 1) * dz_(i + 1)/dz_i'''
        # (None, r) * (r, l)
        return np.dot(output_grad, self.param_)

    def _param_grad(self, output_grad: ndarray) -> ndarray:
        '''Compute Parameters Gradients dL/dw_(i + 1) = dL/dz_(i + 1) * dz_(i + 1)/dw_(i + 1)'''
        # (None, r).T * (None, l)
        return np.dot(output_grad.T, self.input_)


class BiasAdd(ParamOperation):
    '''Compute Bias Addition'''

    def __init__(self, B: ndarray):
        # self.params_ of shape = (neurons, 1)
        super().__init__(B)

    def _output(self) -> ndarray:
        # (None, neurons) + (neurons, 1).T
        return self.input_ + self.param_.T

    def _input_grad(self, output_grad):
        # (None, neurons) .* (None, neurons) where .* means element-wise product
        return np.ones_like(self.input_) * output_grad

    def _param_grad(self, output_grad):
        # (neurons, 1).T .* (None, neurons)
        param_grad = np.ones_like(self.param_.T) * output_grad
        # (None,neurons)->(1,neurons)->(neurons,1)
        return (np.sum(param_grad, axis=0).reshape(1, -1)).T
