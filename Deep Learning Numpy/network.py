from numpy import ndarray
from layers import Layer
from losses import Loss


class NeuralNetwork(object):
    def __init__(self, loss: Loss, seed: int = 1):
        self.loss = loss
        self.seed = seed
        self.layers = []

    def add(self, layer: Layer):
        if self.seed:
            setattr(layer, 'seed', self.seed)
        self.layers.append(layer)

    def forward(self, x_batch: ndarray) -> ndarray:
        x_out = x_batch
        for layer in self.layers:
            x_out = layer.forward(x_out)
        return x_out

    def backward(self, loss_grad: ndarray):
        grad = loss_grad
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return None

    def train(self, x_batch: ndarray, y_batch: ndarray):
        predictions = self.forward(x_batch)
        loss = self.loss.forward(predictions, y_batch)
        self.backward(self.loss.backward())
        return loss

    def params(self):
        for layer in self.layers:
            yield from layer.params

    def param_grads(self):
        for layer in self.layers:
            yield from layer.param_grads
