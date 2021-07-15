import numpy as np
from copy import deepcopy
from numpy import ndarray
from utils import assert_same_shape, permute_data
from network.neural_network import NeuralNetwork
from optimizers.optimizer import Optimizer


class Trainer(object):
    def __init__(self, net: NeuralNetwork, optim: Optimizer, verbose: bool = False):
        self.net = net
        self.optim = optim
        self.best_loss = 1e9
        self.verbose = verbose
        setattr(self.optim, 'net', self.net)

    def generate_batches(
        self, X: ndarray, y: ndarray, batch_size: int = 32
    ) -> tuple[ndarray]:
        assert (
            X.shape[0] == y.shape[0]
        ), '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(
            X.shape[0], y.shape[0]
        )
        N = X.shape[0]
        for ii in range(0, N, batch_size):
            X_batch, y_batch = X[ii : ii + batch_size], y[ii : ii + batch_size]
            yield X_batch, y_batch

    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray,
        y_test: ndarray,
        epochs: int = 100,
        eval_every: int = 10,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
    ):
        np.random.seed(seed)
        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9
        for e in range(1, epochs + 1):
            if e % eval_every == 0:
                last_model = deepcopy(self.net)
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)
            for ii, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train(X_batch, y_batch)
                self.optim.step()
            if e % eval_every == 0:
                test_preds = self.net.forward(X_test)
                loss = self.net.loss.forward(test_preds, y_test)
                if loss < self.best_loss:
                    if self.verbose:
                        print(f"Validation loss after {e} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    if self.verbose:
                        print(
                            f"""Loss increased after epoch {e}, final loss was {self.best_loss:.3f}, using the model from epoch {e-eval_every}"""
                        )
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break

    def predict(self, X: ndarray) -> ndarray:
        return self.net.forward(X)
