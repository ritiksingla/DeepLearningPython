from numpy import ndarray
import sys

sys.path.append("..")
from utils import assert_same_shape
from base import Operation, ParamOperation


class Layer(object):
    def __init__(self, neurons: int):
        self.neurons = neurons
        self.first = True
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []

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
