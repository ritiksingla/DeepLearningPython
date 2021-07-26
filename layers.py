import numpy as np
from numpy import ndarray
from utils import assert_same_shape
from base import (
    Operation,
    ParamOperation,
    WeightMultiply,
    BiasAdd,
    WeightMultiplyElementWise,
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
        self.params: List[ndarray] = []
        self.operations: List[Operation] = []
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
        if self.seed is not None:
            np.random.seed(self.seed)
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


# _________Convolutional Layers_________


class Flatten(Operation):
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

    flatten : bool
        whether to flatten the result (true) or not (false)

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
        self,
        filters: int,
        kernel_size,
        activation: str = 'linear',
        flatten: bool = False,
    ):
        # filters are C_out or units
        super().__init__(filters, activation)
        self.kernel_size = kernel_size
        self.flatten = flatten

    def _setup_layer(self, input_: ndarray) -> ndarray:
        '''
        input_ of size=(batch_size, C_in, H, W)
        '''
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
        if self.flatten:
            self.operations.append(Flatten())
        return None
