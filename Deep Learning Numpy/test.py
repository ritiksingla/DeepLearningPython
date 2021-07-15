import numpy as np
from numpy import ndarray
from layers.dense import Dense
from network.neural_network import NeuralNetwork
from losses.mse_loss import MSELoss
from optimizers.optimizer import Optimizer
from optimizers.sgd import SGD
from activations.linear import Linear
from activations.sigmoid import Sigmoid
from trainer import Trainer
from sklearn.metrics import r2_score

lr = NeuralNetwork(
    layers=[Dense(neurons=1, activation=Linear())], loss=MSELoss(), seed=20190501
)

nn = NeuralNetwork(
    layers=[
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=1, activation=Linear()),
    ],
    loss=MSELoss(),
    seed=20190501,
)

dnn = NeuralNetwork(
    layers=[
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=13, activation=Sigmoid()),
        Dense(neurons=1, activation=Linear()),
    ],
    loss=MSELoss(),
    seed=20190501,
)

from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

from sklearn.preprocessing import StandardScaler

s = StandardScaler()
data = s.fit_transform(data)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=80718
)
y_train = y_train.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# _________No Hidden Layer_________
print("_________No Hidden Layer_________")
trainer = Trainer(lr, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test, epochs=50, eval_every=10, seed=20190501)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)
# _________Single Hidden Layer_________
print("_________Single Hidden Layer_________")
trainer = Trainer(nn, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test, epochs=50, eval_every=10, seed=20190501)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)

# _________2 Hidden Layers_________
print("_________2 Hidden Layers_________")
trainer = Trainer(dnn, SGD(lr=0.01))

trainer.fit(X_train, y_train, X_test, y_test, epochs=50, eval_every=10, seed=20190501)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)
