import numpy as np
from numpy import ndarray
from layers.layer import Layer
from activations.linear import Linear
from base import Operation, WeightMultiply, BiasAdd


class Dense(Layer):
    def __init__(self, neurons: int, activation: Operation = Linear()):
        super().__init__(neurons)
        self.activation = activation

    def _setup_layer(self, input_: ndarray):
        if self.seed is not None:
            np.random.seed(self.seed)
        self.params = []
        # Weights
        self.params.append(np.random.randn(self.neurons, input_.shape[1]))
        # Bias
        self.params.append(np.random.randn(self.neurons, 1))

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]
        return None
