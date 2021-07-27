from layers import Dense, Conv2D, Flatten, MaxPool2D
from network import NeuralNetwork
from optimizers import SGD, RMSProp, Adam
from trainer import Trainer
from lr_schedulers import ReduceLROnPlateau, ExponentialLR

from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Should take around 2-4 minutes!
print("_________Convolutional Neural Network Classification_________")
X, y = load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)
X_train = X_train.reshape(-1, 1, 8, 8)
X_test = X_test.reshape(-1, 1, 8, 8)
model = NeuralNetwork(loss='SoftmaxCrossEntropy', seed=20190119)
model.add(Conv2D(filters=2, kernel_size=3))
model.add(MaxPool2D(pool_size=(2, 2), stride=(1, 1)))
model.add(Flatten())
model.add(Dense(10))
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
