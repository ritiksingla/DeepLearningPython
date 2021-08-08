import numpy as np
from numpy import ndarray
from utils import assert_same_shape
from base import (
    Operation,
    ParamOperation,
    WeightMultiply,
    BiasAdd,
    WeightMultiplyElementWise,
    EmbeddingMultiply,
)
from activations import ACTIVATION_FUNCTIONS
from convolutions import Conv2D_OP


class Layer(object):
    '''
    Base implementation for Layers

    Parameters
    ----------
    units : int
        number of neurons in layer
        (default to 0 is only for BatchNormalization, Dropout, etc.)
        Note : Must be set to positive value before forwarding

    activation : str
        activation function used after applying layer

    Attributes
    ----------
    first : bool
        set to true for initializing attributes first time forward is
        called that depends on input to layer

    params : List[ndarray]
        length = Number of learnable operations from base.py
        It's ith element contains learned parameters for ith operation in layer

    operations : List[Operation]
        length = Number of (learnable + non-learnable(mostly activations))
        operations from base.py

    Warning: This class should not be used directly.
    Use derived classes instead.
    '''

    def __init__(self, units: int = 0, activation: str = 'linear'):
        self.units = units
        self.first = True
        if activation not in ACTIVATION_FUNCTIONS.keys():
            raise ValueError(f'Activation {activation} is not supported currently')
        self.activation = ACTIVATION_FUNCTIONS[activation]()

    def _setup_layer(self, input_: ndarray):
        '''
        Setup self.seed, self.operations list, self.params list to define layer
        at the time input_ provided for first time in self.forward()
        self.seed is setup in network.py
        '''
        raise NotImplementedError()

    def forward(self, input_: ndarray, training: bool = True) -> ndarray:
        '''
        Forward Propagation through the layer.

        Parameters
        ----------
        input_ : numpy array
            size = (None, n_features) if (previous layer is input layer) else (None, prev_units)
            Output from previous layer

        training : bool
            If the current forward is done in training mode (true) or evaluation mode (false)

        Attributes
        ----------
        self.output_ : numpy array
            size = (None, self.units)
            Output from last operation when propagated through self.operations list
            Required only for checking gradients size while back propagating!!

        Returns
        ----------
        self.output_
        '''
        if self.first == True:
            self._setup_layer(input_)
            self.first = False
        assert self.units > 0, f'Neurons must be positive quantity but got {self.units}'

        for operation in self.operations:
            input_ = operation.forward(input_)
        self.output_ = input_
        return self.output_

    def backward(self, output_grad: ndarray) -> ndarray:
        '''
        Back Propagation for computing gradients of operations in operations list
        Non learnable operations computes gradients w.r.t. inputs only whereas
        learnable operations computes gradients w.r.t. params also.

        Parameters
        ----------
        output_grad : numpy array
            size = (None, self.units)
            Gradients w.r.t inputs from (i + 1)th layer

        Returns
        ----------
        output_grad : numpy array
            size = (None, prev_units)
            Gradients w.r.t inputs to (i - 1)th layer
        '''
        assert_same_shape(self.output_, output_grad)

        for operation in reversed(self.operations):
            output_grad = operation.backward(output_grad)
        self._param_grads()
        return output_grad

    def _param_grads(self):
        '''
        Attributes
        ----------
        param_grads : List[ndarray]
            It's ith element contain gradients w.r.t. learnable parameters
            in ith learnable operations (ParamOperation)
        '''
        self.param_grads = []
        for operation in self.operations:
            if issubclass(operation.__class__, ParamOperation):
                self.param_grads.append(operation.param_grad_)

    def __str__(self):
        raise NotImplementedError()


class Dense(Layer):
    '''
    Dense Layer for Neural Network

    Parameters
    ----------
    units : int
        number of neurons in layer
        (default to 0 is only for BatchNormalization, Dropout, etc.)
        Note : Must be set to positive value before forwarding

    use_bias : bool
        Weather to use bias addition (true) or not (false)

    activation : str
        activation function used after applying layer

    Attributes
    ----------
    first : bool
        set to true for initializing attributes first time forward is
        called that depends on input to layer

    params : List[ndarray]
        length = Number of learnable operations from base.py
        It's ith element contains learned parameters for ith operation in layer

    operations : List[Operation]
        length = Number of (learnable + non-learnable(mostly activations))
        operations from base.py
    '''

    def __init__(self, units: int, use_bias: bool = True, activation: str = 'linear'):
        super().__init__(units, activation)
        self.use_bias = use_bias

    def _setup_layer(self, input_: ndarray):
        '''
        Setup self.seed, self.operations list, self.params list to define layer
        at the time input_ provided for first time in self.forward()
        self.seed is setup in network.py
        '''
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []
        # if self.seed is not None:
        #     np.random.seed(self.seed)
        # Weights
        self.params.append(np.random.randn(self.units, input_.shape[1]))

        if self.use_bias == False:
            self.operations = [WeightMultiply(self.params[0]), self.activation]
            return None

        # Bias
        self.params.append(np.random.randn(self.units, 1))

        self.operations = [
            WeightMultiply(self.params[0]),
            BiasAdd(self.params[1]),
            self.activation,
        ]
        return None

    def __str__(self):
        res = str()
        res += (
            f'Dense(units={self.units}, use_bias={self.use_bias},'
            f' activation={self.activation})'
        )
        return res


class BatchNormalization(Layer):
    '''
    Batch Normalization layer for Normalizing input to hidden layers.
    Number of neurons are same as in previous layer.

    Parameters
    ----------
    momentum : float int the range [0, 1]
        controls proportion of history of batches used to normalize inputs

    epsilon : float
        value added to standard deviation of input to avoid 'divide by zero' error

    activation : str
        activation function used after applying layer

    Attributes
    ----------
    first : bool
        set to true for initializing attributes first time forward is
        called that depends on input to layer

    params : List[ndarray]
        length = Number of learnable operations from base.py
        It's ith element contains learned parameters for ith operation in layer

    operations : List[Operation]
        length = Number of (learnable + non-learnable(mostly activations))
        operations from base.py
    '''

    def __init__(
        self, momentum: float = 0.99, epsilon: float = 0.001, activation: str = 'linear'
    ):
        super().__init__(activation=activation)
        self.momentum = momentum
        self.epsilon = epsilon

    def _setup_layer(self, input_: ndarray):
        '''
        Setup self.seed, self.operations list, self.params list to define layer
        at the time input_ provided for first time in self.forward()
        self.seed is setup in network.py

        Attributes
        ----------
        moving_mean : numpy array
            size = (self.units,)
            moving mean of each feature over batches seen during training,
            weight for present mean and previous means depends upon momentum

        moving_var : numpy array
            size = (self.units,)
            moving variance of each feature over batches seen during training,
            weight for present variance and previous variances depends upon momentum
        '''
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []
        if self.seed is not None:
            np.random.seed(self.seed)
        self.units = input_.shape[1]

        self.moving_mean = np.zeros(self.units)

        self.moving_var = np.ones(self.units)

        # Gammas
        self.params.append(np.random.randn(self.units, 1))

        # Betas
        self.params.append(np.random.randn(self.units, 1))

        self.operations = [
            WeightMultiplyElementWise(self.params[0]),
            BiasAdd(self.params[1]),
        ]
        return None

    def forward(self, input_: ndarray, training: bool = True) -> ndarray:
        '''
        Forward Propagation through the layer.
        Output depends upon training mode (training or evaluating).

        moving_mean and moving_var are only updated in training mode.

        moving_mean and moving_var is used for evaluating,
        whereas mean and var only of current batch is used for training.

        Parameters
        ----------
        input_ : numpy array
            size = (None, n_features) if (previous layer is input layer) else (None, prev_units)
            Output from previous layer

        training : bool
            If the current forward is done in training mode (true) or evaluation mode (false)

        Attributes
        ----------
        self.output_ : numpy array
            size = (None, self.units)
            Output from last operation when propagated through self.operations list
            Required only for checking gradients size while back propagating!!

        Returns
        ----------
        self.output_
        '''
        if self.first == True:
            self._setup_layer(input_)
            self.first = False
        if training == True:
            mean_inp = input_.mean(axis=0, keepdims=True)
            var_inp = np.var(input_, axis=0, keepdims=True)
            self.moving_mean = self.moving_mean * self.momentum + mean_inp * (
                1 - self.momentum
            )
            self.moving_var = self.moving_var * self.momentum + var_inp * (
                1 - self.momentum
            )
            input_normalized = (input_ - mean_inp) / (np.sqrt(var_inp + self.epsilon))
        else:
            input_normalized = (input_ - self.moving_mean) / (
                np.sqrt(self.moving_var + self.epsilon)
            )

        for operation in self.operations:
            input_normalized = operation.forward(input_normalized)
        self.output_ = input_normalized
        return self.output_

    def __str__(self):
        res = str()
        res += (
            f'BatchNormalization(momentum={self.momentum}, epsilon={self.epsilon},'
            f' activation={self.activation})\n'
        )
        return res


# _________Convolutional Layers_________


class _flatten(Operation):
    '''
    Flatten the 2D volume of convolutional layers

    Parameters
    ----------
    input_ : ndarray of size = (batch_size, channels, H, W)

    Returns
    ----------
    output_ : ndarray of size = (batch_size, channels * H * W)
    '''

    def __init__(self):
        super().__init__()

    def _output(self) -> ndarray:
        return self.input_.reshape(self.input_.shape[0], -1)

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Parameters
        ----------
        output_grad : ndarray of size = (batch_size, channels * H * W)

        Returns
        ----------
        View of output_grad : ndarray of size = (batch_size, channels, H, W)
        '''
        return output_grad.reshape(self.input_.shape)


class Flatten(Layer):
    '''
    Wrapper over _flatten layer

    Parameters
    ----------
    input_ : ndarray of size = (batch_size, channels, H, W)

    Returns
    ----------
    output_ : ndarray of size = (batch_size, channels * H * W)
    '''

    def __init__(self, activation: str = 'linear'):
        super().__init__(activation)

    def _setup_layer(self, input_):
        self.operations: List[Operation] = []
        self.units = input_.shape[1]
        self.operations.append(_flatten())

    def __str__(self):
        res = str()
        res += f'Flatten()\n'
        return res


class _max_pool2d(Operation):
    '''
    Max Pools the 2D volume of convolutional layers

    Parameters
    ----------
    input_ : ndarray of size = (batch_size, channels, H, W)

    Returns
    ----------
    output_ : ndarray of size = (
                                batch_size,
                                channels,
                                ((H - pool_size[0]) // stride[0]) + 1,
                                ((W - pool_size[1]) // stride[1]) + 1,
                                )
    '''

    def __init__(self, pool_size, stride):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def _output(self):
        output_ = np.zeros(
            (
                self.input_.shape[0],
                self.input_.shape[1],
                (self.input_.shape[2] - self.pool_size[0]) // self.stride[0] + 1,
                (self.input_.shape[3] - self.pool_size[1]) // self.stride[1] + 1,
            )
        )
        for i in range(0, self.input_.shape[2] - self.pool_size[0] + 1, self.stride[0]):
            for j in range(
                0, self.input_.shape[3] - self.pool_size[1] + 1, self.stride[1]
            ):
                output_[:, :, (i // self.stride[0]), (j // self.stride[1])] = np.max(
                    self.input_[
                        :, :, i : i + self.pool_size[0], j : j + self.pool_size[1]
                    ],
                    axis=(2, 3),
                )
        return output_

    def _input_grad(self, output_grad: ndarray) -> ndarray:
        '''
        Parameters
        ----------
        output_grad : ndarray of size = (
                                batch_size,
                                channels,
                                ((H - pool_size[0]) // stride[0]) + 1,
                                ((W - pool_size[1]) // stride[1]) + 1,
                                )

        Returns
        ----------
        View of output_grad : ndarray of size = (batch_size, channels, H, W)
        '''
        input_grad = np.zeros_like(self.input_)
        for batch_i in range(self.input_.shape[0]):
            for c in range(self.input_.shape[1]):
                for i in range(
                    0, self.input_.shape[2] - self.pool_size[0] + 1, self.stride[0]
                ):
                    for j in range(
                        0, self.input_.shape[3] - self.pool_size[1] + 1, self.stride[1]
                    ):
                        maxi = self.output_[
                            batch_i, c, (i // self.stride[0]), (j // self.stride[1])
                        ]
                        max_mask = np.where(
                            self.input_[
                                batch_i,
                                c,
                                i : i + self.pool_size[0],
                                j : j + self.pool_size[1],
                            ]
                            == maxi,
                            1,
                            0,
                        )
                        input_grad[
                            batch_i,
                            c,
                            i : i + self.pool_size[0],
                            j : j + self.pool_size[1],
                        ] += (
                            max_mask
                            * output_grad[
                                batch_i, c, (i // self.stride[0]), (j // self.stride[1])
                            ]
                        )
        return input_grad


class MaxPool2D(Layer):
    '''
    Wrapper over _max_pool2d layer
    '''

    def __init__(self, pool_size: tuple, stride: tuple, activation: str = 'linear'):
        super().__init__(activation)
        self.pool_size = pool_size
        self.stride = stride

    def _setup_layer(self, input_):
        self.operations: List[Operation] = []
        self.units = 1
        self.operations.append(_max_pool2d(self.pool_size, self.stride))

    def __str__(self):
        res = str()
        res += f'MaxPool()\n'
        return res


class Conv2D(Layer):
    '''
    2D Convolution layer

    Parameters
    ----------
    filters : int
        Number of filters or output channels (depth of volume)

    kernel_size : int
        Shape of kernel to apply (kernel_size, kernel_size)

    activation : str
        activation function used after applying layer

    Attributes
    ----------
    first : bool
        set to true for initializing attributes first time forward is
        called that depends on input to layer

    params : List[ndarray]
        length = Number of learnable operations from base.py
        It's ith element contains learned parameters for ith operation in layer

    operations : List[Operation]
        length = Number of (learnable + non-learnable(mostly activations))
        operations from base.py
    '''

    def __init__(self, filters: int, kernel_size, activation: str = 'linear'):
        # filters are C_out or units
        super().__init__(filters, activation)
        self.kernel_size = kernel_size

    def _setup_layer(self, input_: ndarray) -> ndarray:
        '''
        input_ of size=(batch_size, C_in, H, W)
        '''
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []
        if self.seed is not None:
            np.random.seed(self.seed)
        in_channels = input_.shape[1]
        out_channels = self.units
        conv_params = np.random.normal(
            loc=0,
            scale=1.0,
            size=(out_channels, in_channels, self.kernel_size, self.kernel_size),
        )
        self.params.append(conv_params)
        self.operations = [Conv2D_OP(conv_params), self.activation]
        return None

    def __str__(self):
        res = str()
        res += (
            f'Conv2D(filters={self.units}, kernel_size={self.kernel_size},'
            f' activation={self.activation})\n'
        )
        return res


class Embedding(Layer):
    '''
    input shape=(batch_size, sequence_length) [with each number between [0, num_embeddings) ]
    output shape=(batch_size, sequence_length, embedding_dim)
    '''

    def __init__(
        self, num_embeddings: int, embedding_dim: int, activation: str = 'linear'
    ):
        super().__init__(activation)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

    def _setup_layer(self, input_):
        self.operations: List[Operation] = []
        self.params: List[ndarray] = []
        self.units = 1
        self.params.append(np.random.randn(self.embedding_dim, self.num_embeddings))

        self.operations = [EmbeddingMultiply(self.params[0])]

    def __str__(self):
        res = str()
        res += f'Embedding()\n'
        return res
