from numpy import ndarray
from layers import Layer
from losses import LOSS_FUNCTIONS


class NeuralNetwork(object):
    '''
    Class for training Neural Network

    Parameters
    ----------
    loss : str
        Loss function for output
    seed : int
        seed for random

    Attributes
    ----------
    layers : list[Layer]
        list of layers in sequential
    '''

    def __init__(self, loss: str, seed: int = 1):
        if loss not in LOSS_FUNCTIONS.keys():
            raise ValueError(f'Loss {loss} is not supported currently')
        self.loss = LOSS_FUNCTIONS[loss]()
        self.seed = seed
        self.layers = []

    def add(self, layer: Layer):
        '''
        Adds the layer to the model in sequential manner
        '''
        if self.seed:
            setattr(layer, 'seed', self.seed)
        self.layers.append(layer)

    def forward(self, x_batch: ndarray, training: bool = True) -> ndarray:
        '''
        Forward Propagation for current batch through whole network

        Parameters
        ----------
        x_batch : ndarray
            current batch for forward propagation
        training : bool
            whether training is in training mode or evaluation mode

        Returns
        ----------
        x_out : ndarray
            output from last layer when propagated through self.layers list
        '''
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out, training=training)
        return x_out

    def backward(self, loss_grad: ndarray):
        '''
        Back propagation through whole network

        Parameters
        ----------
        loss_grad : ndarray
            Gradients w.r.t. inputs from 'Loss' class
        '''
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def train(self, x_batch: ndarray, y_batch: ndarray):
        '''
        Train single batch of input and target
        '''
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        '''
        Returns iterator to layer parameters
        Used for updating parameters in optimization step
        '''
        for layer in self.layers:
            if not hasattr(layer, 'params'):
                continue
            yield from layer.params

    def param_grads(self):
        '''
        Returns iterator to layer parameter's gradients : dL / d_params
        Used for updating parameters in optimization step
        '''
        for layer in self.layers:
            yield from layer.param_grads

    def __str__(self):
        res = str()
        for layer in self.layers:
            res += str(layer)
        return res
