from layers import Dense, BatchNormalization
from network import NeuralNetwork
from optimizers import SGD, RMSProp, Adam
from trainer import Trainer
from lr_schedulers import ReduceLROnPlateau, ExponentialLR

from sklearn.metrics import r2_score, accuracy_score
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification


boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

s = StandardScaler()
data = s.fit_transform(data)
target = target.reshape(-1, 1)

X_train, X_test, y_train, y_test = train_test_split(
    data, target, test_size=0.3, random_state=80718
)
# _________No Hidden Layer Regression_________
print("_________No Hidden Layer_________")

lr = NeuralNetwork(loss='MSELoss', seed=20190501)
lr.add(Dense(units=1))

trainer = Trainer(net=lr, optim=SGD(lr=0.01), verbose=True)

trainer.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10, seed=20190501)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)

# _________Single Hidden Layer Regression_________
print("_________Single Hidden Layer_________")

nn = NeuralNetwork(loss='MSELoss', seed=20190501)
nn.add(Dense(units=13, activation='sigmoid'))
nn.add(Dense(units=1))
optimizer = Adam(lr=0.1)
trainer = Trainer(net=nn, optim=optimizer, verbose=True)

trainer.fit(X_train, y_train, X_test, y_test, epochs=10, eval_every=10, seed=20190501)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)

# _________2 Hidden Layers Regression_________
print("_________2 Hidden Layers_________")

dnn = NeuralNetwork(loss='MSELoss', seed=20190501)
dnn.add(Dense(units=13, use_bias=False))
dnn.add(BatchNormalization())
dnn.add(Dense(units=13, activation='sigmoid'))
dnn.add(Dense(units=1))

trainer = Trainer(net=dnn, optim=Adam(lr=0.01, nesterov=True), verbose=True)

# Using large batch size for BatchNormalization layer
trainer.fit(
    X_train,
    y_train,
    X_test,
    y_test,
    eval_every=10,
    batch_size=128,
    epochs=200,
    seed=20190501,
)

print(
    'R2 Score for training data: {:.2f}'.format(
        r2_score(y_train, trainer.predict(X_train))
    )
)

print(
    'R2 Score for test data: {:.2f}'.format(r2_score(y_test, trainer.predict(X_test)))
)

# _________Classification_________
print("_________Classification_________")
print("_________1 Hidden Layer_________")

# Generate a binary classification dataset.
X, y = make_classification(
    n_samples=500, n_features=25, n_clusters_per_class=1, n_classes=4, n_informative=15
)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)

model = NeuralNetwork(loss='SoftmaxCrossEntropy', seed=20190119)
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(4))

trainer = Trainer(model, RMSProp(lr=0.01), classification=True, verbose=True)
trainer.fit(X_train, y_train, X_test, y_test, epochs=150, eval_every=10, seed=20190501)
print(
    'Accuracy Score for training data: {:.2f}'.format(
        accuracy_score(y_train, trainer.predict(X_train))
    )
)

print(
    'Accuracy Score for test data: {:.2f}'.format(
        accuracy_score(y_test, trainer.predict(X_test))
    )
)
