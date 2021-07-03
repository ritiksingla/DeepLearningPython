import numpy as np
import matplotlib.pyplot as plt
from string import punctuation
from collections import Counter
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn

# Loading the data and visualization
with open(r'../input/imdb-dataset/reviews.txt', 'r') as f:
    reviews = f.read()
with open(r'../input/imdb-dataset/labels.txt', 'r') as f:
    labels = f.read()

# Data Preprocessing
reviews = reviews.lower()
all_text = ''.join([c for c in reviews if c not in punctuation])
reviews_split = all_text.split('\n')
all_text = ' '.join(reviews_split)
words = all_text.split()

# Encoding the words
counts = Counter(words)
# Most frequent words get initial indices
vocab = sorted(counts, key=counts.get, reverse=True)
vocab_to_int = {word: idx for idx, word in enumerate(vocab, 1)}

reviews_ints = []
for review in reviews_split:
    reviews_ints.append([vocab_to_int[word] for word in review.split()])

# Encoding the labels
labels_split = labels.split('\n')
encoded_labels = np.array([1 if label == 'positive' else 0 for label in labels_split])

# Removing outliers
"""
1. Get rid of extremely short or long texts
1. Padding/truncating the remaining data so that we have reviews of the same length.
"""

review_lens = Counter([len(x) for x in reviews_ints])
# print("Zero-length reviews: {}".format(review_lens[0]))
# print("Maximum review length: {}".format(max(review_lens)))

non_zero_idx = [idx for idx, review in enumerate(reviews_ints) if len(review) != 0]

reviews_ints = [reviews_ints[idx] for idx in non_zero_idx]
encoded_labels = np.array([encoded_labels[idx] for idx in non_zero_idx])
assert len(reviews_ints) == 25000

# Padding sequences
"""
For example, if the seq_length=10 and an input review is:
[117, 18, 128]

The resultant, padded sequence should be:
[0, 0, 0, 0, 0, 0, 0, 117, 18, 128]

Our final features array will be a 2D array, with as many rows 
as there are reviews, and as many columns as the specified seq_length.
"""


def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for idx, x in enumerate(reviews_ints):
        if len(x) < seq_length:
            features[idx, seq_length - len(x) :] = x
        elif len(x) > seq_length:
            features[idx, :] = x[0:seq_length]
    return features


seq_length = 200
features = pad_features(reviews_ints, seq_length=seq_length)

assert len(features) == len(
    reviews_ints
), "Features should have as many rows as reviews."

assert (
    len(features[0]) == seq_length
), 'Features should have as many columns as seq_length'

# Training, Validation, Test

split_frac = 0.8

split_idx = int(len(features) * split_frac)
train_x, remaining_x = features[:split_idx], features[split_idx:]
train_y, remaining_y = encoded_labels[:split_idx], encoded_labels[split_idx:]

test_idx = int(len(remaining_x) * 0.5)
val_x, test_x = remaining_x[:test_idx], remaining_x[test_idx:]
val_y, test_y = remaining_y[:test_idx], remaining_y[test_idx:]

# First dimension of both features and labels must be same.
train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

batch_size = 50

train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

dataiter = iter(train_loader)
sample_x, sample_y = dataiter.next()

# Define the model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SentimentRNN(nn.Module):
    def __init__(
        self,
        vocab_size,
        output_size,
        embedding_dim,
        hidden_dim,
        n_layers,
        drop_prob=0.5,
    ):
        super().__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim
        )
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=drop_prob,
            batch_first=True,
        )

        self.dropout = nn.Dropout(p=drop_prob)
        self.fc1 = nn.Linear(in_features=hidden_dim, out_features=output_size)
        self.sig = nn.Sigmoid()

    def forward(self, X, Hidden):
        """
        Shape of X will be (batch_size, seq_length)

        Shape of embeds will be (batch_size, seq_length, embedding_dim)

        Shape of lstm_out will be (batch_size, seq_length, hidden_dim)
        and after taking last time-stamp it will be (batch_size, hidden_dim)

        Shape of hidden will be (num_layers, batch_size, hidden_dim)

        Shape of out will be (batch_size, 1)
        """
        batch_size = X.size(0)

        X = X.long()
        embeds = self.embedding(X)
        lstm_out, Hidden = self.lstm(embeds, Hidden)

        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        # dropout and fully-connected layer
        out = self.dropout(lstm_out)
        out = self.fc1(out)

        # sigmoid function
        sig_out = self.sig(out)

        # reshape to be batch_size first
        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]  # get last batch of labels

        # return last sigmoid output and hidden state
        return sig_out, Hidden

    def init_hidden(self, batch_size):
        # initialized to zero, for hidden state and cell state of LSTM

        weight = next(self.parameters()).data

        hidden = (
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
            weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device),
        )

        return hidden


vocab_size = len(vocab_to_int) + 1  # +1 for the 0 padding + our word tokens
output_size = 1
embedding_dim = 400
hidden_dim = 256
n_layers = 2

net = SentimentRNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)
net.to(device)

# Training the model
lr = 0.001
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr)

epochs = 4  # 3-4 is approx where I noticed the validation loss stop decreasing

counter = 0
print_every = 100
clip = 5  # gradient clipping

net.train()
for e in range(epochs):
    # initialize hidden state
    h = net.init_hidden(batch_size)

    # batch loop
    for inputs, labels in train_loader:
        counter += 1

        inputs, labels = inputs.to(device), labels.to(device)

        # Creating new variables for the hidden state, otherwise
        # we'd backprop through the entire training history
        h = tuple([each.data for each in h])

        # zero accumulated gradients
        net.zero_grad()

        # get the output from the model
        output, h = net(inputs, h)

        # calculate the loss and perform backprop
        loss = criterion(output.squeeze(), labels.float())
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        nn.utils.clip_grad_norm_(net.parameters(), clip)
        optimizer.step()

        # loss stats
        if counter % print_every == 0:
            # Get validation loss
            val_h = net.init_hidden(batch_size)
            val_losses = []
            net.eval()
            for inputs, labels in valid_loader:

                # Creating new variables for the hidden state, otherwise
                # we'd backprop through the entire training history
                val_h = tuple([each.data for each in val_h])

                inputs, labels = inputs.to(device), labels.to(device)

                output, val_h = net(inputs, val_h)
                val_loss = criterion(output.squeeze(), labels.float())

                val_losses.append(val_loss.item())

            net.train()
            print(
                "Epoch: {}/{}...".format(e + 1, epochs),
                "Step: {}...".format(counter),
                "Loss: {:.6f}...".format(loss.item()),
                "Val Loss: {:.6f}".format(np.mean(val_losses)),
            )

# Testing the model
test_losses = []  # track loss
num_correct = 0

# init hidden state
h = net.init_hidden(batch_size)

net.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    inputs, labels = inputs.to(device), labels.to(device)

    # get predicted outputs
    output, h = net(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)


# Metrics (Evaluate the model)

# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))

# Inference on a test review
test_review_neg = (
    'The worst movie I have seen; acting was terrible and I want my money back. This'
    ' movie had bad acting and the dialogue was slow.'
)


def tokenize_review(test_review):
    test_review = test_review.lower()
    test_text = ''.join([c for c in test_review if c not in punctuation])
    test_words = test_text.split()
    test_ints = []
    test_ints.append([vocab_to_int[word] for word in test_words])
    return test_ints


test_ints = tokenize_review(test_review_neg)

seq_length = 200
features = pad_features(test_ints, seq_length)

feature_tensor = torch.from_numpy(features)


def predict(net, test_review, sequence_length=200):

    net.eval()

    # tokenize review
    test_ints = tokenize_review(test_review)

    # pad tokenized sequence
    seq_length = sequence_length
    features = pad_features(test_ints, seq_length)

    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(features)

    batch_size = feature_tensor.size(0)

    # initialize hidden state
    h = net.init_hidden(batch_size)

    feature_tensor = feature_tensor.to(device)

    # get the output from the model
    output, h = net(feature_tensor, h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if pred.item() == 1:
        print("Positive review detected!")
    else:
        print("Negative review detected.")


test_review_pos = (
    'This movie had the best acting and the dialogue was so good. I loved it.'
)

seq_length = 200  # good to use the length that was trained on

predict(net, test_review_neg, seq_length)
predict(net, test_review_pos, seq_length)
my_review = "I am happy that I didn't saw this movie."
predict(net, my_review, seq_length)
