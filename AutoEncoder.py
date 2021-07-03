""" With autoencoders, we pass input data through an encoder that makes a compressed
representation of the input. Then, this representation is passed through a decoder to
reconstruct the input data. Generally the encoder and decoder will be built with 
neural networks, then trained on example data."""

# Downloading MNIST Dataset

import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn

transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
test_data = datasets.MNIST(root='data', train=False, download=True, transform=transform)

# Setting Data Loader Parameters and loading data in DataLoaders

BATCH_SIZE = 20
NUM_WORKERS = 0
INPUT_SIZE = 28
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)
test_loader = torch.utils.data.DataLoader(
    test_data, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS
)

# Visualizing Data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

# get one image from the batch
img = np.squeeze(images[0])

fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')

# Defining simple DNN
# The model takes (None, 28 * 28) sized flatten images and encoding dimension for hidden layer size


class AutoEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(AutoEncoder, self).__init__()

        self.fc1 = nn.Linear(INPUT_SIZE * INPUT_SIZE, encoding_dim)

        self.fc2 = nn.Linear(encoding_dim, INPUT_SIZE * INPUT_SIZE)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))

        return x


ENCODING_DIM = 32
model = AutoEncoder(ENCODING_DIM)

# Defining the loss and optimization algorithm
# As we are dealing with pixel-wise loss when comparing output to input

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training the model and calculating loss
n_epochs = 20
for epoch in range(1, n_epochs + 1):
    training_loss = 0.0
    steps_per_epoch = 0
    # We don't care about labels right now
    for X, _ in train_loader:
        # Flatten with getting X.size(0) for batch size (20 in this case)
        X = X.view(X.size(0), -1)
        # Clear the gradients for previous mini-batches
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, X)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        steps_per_epoch += 1
    else:
        print(
            'Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, training_loss / steps_per_epoch
            )
        )

# Testing the quality of encoding, given test images

dataiter = iter(test_loader)
images, labels = dataiter.next()

images_flatten = images.view(images.size(0), -1)
output = model(images_flatten)
images = images.numpy()
output = output.view(BATCH_SIZE, 1, 28, 28)

# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

"""Results: Encoding is quite good except for some blurriness in some parts"""


# AutoEncoding using Upsampling and Convolution Layers

""" As we are dealing with images, it's better to use convolutions to autoencode them.
    To reduce dimensions we will use max-pool + conv2d in encoder and to increase back to 
    original size we will use upsampling using nearest-neighbour technique + conv2d"""

Encoder = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=2, stride=2),
)  # encode (None, 1, 28, 28) images to (None, 4, 7, 7)

Decoder = nn.Sequential(
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(in_channels=4, out_channels=16, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.Conv2d(in_channels=16, out_channels=1, kernel_size=3, padding=1),
    nn.Sigmoid(),
)

model = nn.Sequential(Encoder, Decoder)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5
for epoch in range(1, n_epochs + 1):
    training_loss = 0.0
    steps_per_epoch = 0
    # We don't care about labels right now
    for X, _ in train_loader:
        # Clear the gradients for previous mini-batches
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, X)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        steps_per_epoch += 1
    else:
        print(
            'Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, training_loss / steps_per_epoch
            )
        )

dataiter = iter(test_loader)
images, labels = dataiter.next()

output = model(images)
images = images.numpy()
output = output.view(BATCH_SIZE, 1, 28, 28)

# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

# AutoEncoding using Convolution and Deconvolution(Transpose Convolution)

""" Convolution output size is given by (W + 2P - K)//S + 1
    Deconvolution output size is given by (W - 1) * S + K - 2P
"""

# Encoder remains same as in Upsampling
Decoder = nn.Sequential(
    nn.ConvTranspose2d(in_channels=4, out_channels=16, kernel_size=2, stride=2),
    nn.ReLU(),
    nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=2, stride=2),
    nn.Sigmoid(),
)

model = nn.Sequential(Encoder, Decoder)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

n_epochs = 5
for epoch in range(1, n_epochs + 1):
    training_loss = 0.0
    steps_per_epoch = 0
    # We don't care about labels right now
    for X, _ in train_loader:
        # Clear the gradients for previous mini-batches
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, X)
        loss.backward()
        optimizer.step()
        training_loss += loss.item()
        steps_per_epoch += 1
    else:
        print(
            'Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, training_loss / steps_per_epoch
            )
        )

dataiter = iter(test_loader)
images, labels = dataiter.next()

output = model(images)
images = images.numpy()
output = output.view(BATCH_SIZE, 1, 28, 28)

# use detach when it's an output that requires_grad
output = output.detach().numpy()

# plot the first ten input images and then reconstructed images
fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

# input images on top row, reconstructions on bottom
for images, row in zip([images, output], axes):
    for img, ax in zip(images, row):
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
