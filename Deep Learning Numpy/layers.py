import numpy as np
from numpy import ndarray
from utils import assert_same_shape
from base import Operation, ParamOperation, WeightMultiply, BiasAdd
from activations import Linear, Sigmoid, Tanh, ReLU, Softmax


class Layer(object):
    def __init__(self, units: int, activation: str = 'linear'):
        self.units = units
        self.first = True
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []
        assert activation in (
            'linear',
            'sigmoid',
            'tanh',
            'relu',
            'softmax',
        ), 'activation "{}" is currently not supported'.format(activation)
        if activation == 'linear':
            self.activation = Linear()
        elif activation == 'sigmoid':
            self.activation = Sigmoid()
        elif activation == 'tanh':
            self.activation = Tanh()
        elif activation == 'relu':
            self.activation = ReLU()
        elif activation == 'softmax':
            self.activation = Softmax()

    def _setup_layer(self, num_in: int):
        raise NotImplementedError()

    def forward(self, input_: ndarray) -> ndarray:
        if self.first == True:
            self._setup_layer(input_)
            self.first = False
        self.input_ = input_

        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output_ = input_
        return self.output_

    def backward(self, output_grad: ndarray) -> ndarray:
        assert_same_shape(self.output_, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        input_grad = output_grad
        self._param_grads()
        return input_grad

    def _param_grads(self) -> ndarray:
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad_)

    def _params(self):
        self.params = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.params.append(operation.param_)


class Dense(Layer):
    def __init__(self, units: int, activation: str = 'linear'):
        super().__init__(units, activation)

    def _setup_layer(self, input_: ndarray):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.params = []
        # Weights
        self.params.append(np.random.randn(self.units, input_.shape[1]))
        # Bias
        self.params.append(np.random.randn(self.units, 1))

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]
        return None
