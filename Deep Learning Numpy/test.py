import numpy as np
from layers import Dense
from network import NeuralNetwork
from losses import MSELoss, L1Loss, BCELoss
from optimizers import SGD
from trainer import Trainer
from sklearn.metrics import r2_score, accuracy_score

criterion = MSELoss()
lr = NeuralNetwork(loss=criterion, seed=20190501)
lr.add(Dense(units=1))

criterion = MSELoss()
nn = NeuralNetwork(loss=criterion, seed=20190501)
nn.add(Dense(units=13, activation='sigmoid'))
nn.add(Dense(units=1))

criterion = MSELoss()
dnn = NeuralNetwork(loss=criterion, seed=20190501)
dnn.add(Dense(units=13, activation='sigmoid'))
dnn.add(Dense(units=13, activation='sigmoid'))
dnn.add(Dense(units=1))

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

# _________No Hidden Layer Regression_________
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

# _________Single Hidden Layer Regression_________
print("_________Single Hidden Layer_________")
trainer = Trainer(nn, SGD(lr=0.01, momentum=0.9, dampening=0))

trainer.fit(X_train, y_train, X_test, y_test, epochs=100, eval_every=10, seed=20190501)

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


# _________Classification_________
print("_________Single Hidden Layer Classification_________")
from sklearn.datasets import make_classification

# Generate a binary classification dataset.
X, y = make_classification(
    n_samples=500, n_features=25, n_clusters_per_class=1, n_classes=4, n_informative=15
)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
model = NeuralNetwork(loss=BCELoss(), seed=20190119)
model.add(Dense(10, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))

trainer = Trainer(model, SGD(lr=0.1, momentum=0.9, dampening=0), classification=True)
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
