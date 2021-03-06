import numpy as np
from numpy import ndarray
from utils import assert_same_shape, permute_data
from network import NeuralNetwork
from optimizers import Optimizer

from copy import deepcopy
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, r2_score


class Trainer(object):
    def __init__(
        self,
        net: NeuralNetwork,
        optim: Optimizer,
        lr_schedule=None,
        classification: bool = False,
        verbose: bool = False,
    ):
        self.net = net
        self.optim = optim
        self.lr_schedule = lr_schedule
        self.best_loss = 1e9
        self.verbose = verbose
        self.classification = classification
        setattr(self.optim, 'net', self.net)

    def generate_batches(
        self, X: ndarray, y: ndarray, batch_size: int = 32
    ) -> tuple[ndarray]:
        '''
        Generate batches according to batch_size parameter passed
        '''
        assert (
            X.shape[0] == y.shape[0]
        ), '''
        features and target must have the same number of rows, instead
        features has {0} and target has {1}
        '''.format(
            X.shape[0], y.shape[0]
        )
        N = X.shape[0]
        for i_batch in range(0, N, batch_size):
            X_batch, y_batch = (
                X[i_batch : i_batch + batch_size],
                y[i_batch : i_batch + batch_size],
            )
            yield X_batch, y_batch

    @property
    def get_metrics_max(self):
        if self.classification:
            return accuracy_score
        else:
            return r2_score

    def fit(
        self,
        X_train: ndarray,
        y_train: ndarray,
        X_test: ndarray = None,
        y_test: ndarray = None,
        epochs: int = 100,
        eval_every: int = None,
        batch_size: int = 32,
        seed: int = 1,
        restart: bool = True,
    ):
        if eval_every is not None:
            assert X_test is not None and y_test is not None
        np.random.seed(seed)
        assert len(self.net.layers) != 0, 'add layers to train the model'

        if restart:
            for layer in self.net.layers:
                layer.first = True
            self.best_loss = 1e9
        if self.classification:
            lb = LabelBinarizer()
            y_train = lb.fit_transform(y_train)
            if eval_every is not None:
                y_test = lb.transform(y_test)

        for epoch in range(1, epochs + 1):
            if eval_every is not None and epoch % eval_every == 0:
                last_model = deepcopy(self.net)
            X_train, y_train = permute_data(X_train, y_train)
            batch_generator = self.generate_batches(X_train, y_train, batch_size)

            # Training Loop
            for i_batch, (X_batch, y_batch) in enumerate(batch_generator):
                self.net.train(X_batch, y_batch)
                # Take optimization step
                self.optim.step(i_batch + 1)
            # Perform LR Decay if any
            if self.lr_schedule is not None:
                self.lr_schedule.forward(epoch)

            # Evaluation Block
            if eval_every is not None and epoch % eval_every == 0:
                test_preds = self.net.forward(X_test, training=False)
                loss = self.net.loss.forward(test_preds, y_test)
                if loss < self.best_loss:
                    if self.verbose:
                        print(f"Validation loss after {epoch} epochs is {loss:.3f}")
                    self.best_loss = loss
                else:
                    if self.verbose:
                        print(
                            f"""Loss increased after epoch {epoch}, final loss was {self.best_loss:.3f}, using the model from epoch {epoch-eval_every}"""
                        )
                    self.net = last_model
                    # ensure self.optim is still updating self.net
                    setattr(self.optim, 'net', self.net)
                    break

    def predict(self, X: ndarray) -> ndarray:
        '''
        Predicts the target for input.
        Training = False is useful for (BatchNorm, Dropout) like layers which behave
        differently for prediction
        '''
        pred = self.net.forward(X, training=False)
        if self.classification:
            return np.argsort(pred, axis=1)[:, -1]
        return pred
