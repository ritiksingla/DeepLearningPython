import numpy as np
from numpy import ndarray
from layers import Dense, BatchNormalization, Flatten, Embedding
from optimizers import SGD, RMSProp, Adam
from recurrent import RecurrentNetwork, RNN


def get_batch(a: ndarray, epoch: int, batch_size: int):
    return a[epoch * batch_size : (epoch + 1) * batch_size]


# Define the model

vocab_size = 10000
output_size = 2
embedding_dim = 400
hidden_dim = 256
batch_size = 32

rnn = RecurrentNetwork(loss='SoftmaxCrossEntropy')
rnn.add(Embedding(vocab_size, embedding_dim))
rnn.add(RNN(embedding_dim, hidden_dim))
rnn.add(Flatten())
rnn.add(Dense(output_size))
optim = Adam(lr=0.001)
setattr(optim, 'net', rnn)
total_epochs = 8
train_x = np.random.randint(0, vocab_size, size=(total_epochs * batch_size, 10))
train_y = np.random.randint(0, output_size, size=(total_epochs * batch_size, 1))
for epoch in range(total_epochs):
    batch_x = get_batch(train_x, epoch, batch_size)
    batch_y = get_batch(train_y, epoch, batch_size)
    loss = rnn.train(batch_x, batch_y)
    print(loss)
    optim.step(epoch + 1)

from sklearn.metrics import r2_score, accuracy_score

batch_x = get_batch(train_x, 1, batch_size)
batch_y = get_batch(train_y, 1, batch_size)
print(
    'Accuracy Score for training data: {:.2f}'.format(
        accuracy_score(batch_y, rnn.predict(batch_x))
    )
)
