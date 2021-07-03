import numpy as np
import torch
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

num_workers = 2
batch_size = 64
transform = transforms.ToTensor()
train_data = datasets.MNIST(root='data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers
)

dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy()

img = np.squeeze(images[0])

fig = plt.figure(figsize=(3, 3))
ax = fig.add_subplot(111)
ax.imshow(img, cmap='gray')


class Generator(nn.Module):
    def __init__(self, latent_vector_size, hidden_size, img_size):
        super(Generator, self).__init__()

        self.fc1 = nn.Linear(in_features=latent_vector_size, out_features=hidden_size)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=hidden_size * 2)
        self.fc3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size * 4)

        self.fc4 = nn.Linear(in_features=4 * hidden_size, out_features=img_size)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = F.leaky_relu(input=self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(input=self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(input=self.fc3(x), negative_slope=0.2)
        x = self.dropout(x)
        out = torch.tanh(input=self.fc4(x))
        return out


class Discriminator(nn.Module):
    def __init__(self, img_size, hidden_size):
        super(Discriminator, self).__init__()

        self.fc1 = nn.Linear(in_features=img_size, out_features=4 * hidden_size)
        self.fc2 = nn.Linear(in_features=4 * hidden_size, out_features=hidden_size * 2)
        self.fc3 = nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)

        self.fc4 = nn.Linear(in_features=hidden_size, out_features=1)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.leaky_relu(input=self.fc1(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(input=self.fc2(x), negative_slope=0.2)
        x = self.dropout(x)
        x = F.leaky_relu(input=self.fc3(x), negative_slope=0.2)
        x = self.dropout(x)
        out = torch.sigmoid(input=self.fc4(x))
        return out


img_size = 784
hidden_size = 32
latent_vector_size = 100

G = Generator(latent_vector_size, hidden_size, img_size)
D = Discriminator(img_size, hidden_size)

# output 1 for real
def real_loss(D_out, smooth=False):
    batch_size = D_out.size(0)
    # smooth is true if we don't want discriminator to train very quickly
    if smooth:
        labels = torch.ones(batch_size) * 0.9
    else:
        labels = torch.ones(batch_size)

    criterion = nn.BCELoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


# output 0 for fake
def fake_loss(D_out):
    batch_size = D_out.size(0)
    labels = torch.zeros(batch_size)
    criterion = nn.BCELoss()
    loss = criterion(D_out.squeeze(), labels)
    return loss


import torch.optim as optim

lr = 0.002
d_optimizer = optim.Adam(D.parameters(), lr)
g_optimizer = optim.Adam(G.parameters(), lr)

import pickle as pkl

num_epochs = 100

samples = []
losses = []

print_every = 400

# Get some fixed data for sampling. These are images that are held
# constant throughout training, and allow us to inspect the model's performance
sample_size = 16
fixed_z = np.random.uniform(-1, 1, size=(sample_size, latent_vector_size))
fixed_z = torch.from_numpy(fixed_z).float()

D.train()
G.train()
for epoch in range(num_epochs):
    for batch_i, (real_images, _) in enumerate(train_loader):
        batch_size = real_images.size(0)
        # Rescaling images from [0, 1) to [-1, 1) as we used tanh in gen output and disc input
        real_images = real_images * 2 - 1

        # 1. Training Discriminator
        d_optimizer.zero_grad()

        # 1.1 Calculating real loss
        D_real = D(real_images)
        # Penalize perfect guessing also by providing smooth = True
        d_real_loss = real_loss(D_real, smooth=True)

        # 1.2 Calculating fake loss
        with torch.no_grad():  # As we don't want to accumulate generator gradients right now!
            z = np.random.uniform(-1, 1, size=(batch_size, latent_vector_size))
            z = torch.from_numpy(z).float()
            fake_images = G(z)
        D_fake = D(fake_images)
        d_fake_loss = fake_loss(D_fake)

        # 1.3 Sum up real and fake loss for disc
        d_loss = d_real_loss + d_fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 2. Training Generator
        g_optimizer.zero_grad()
        z = np.random.uniform(-1, 1, size=(batch_size, latent_vector_size))
        z = torch.from_numpy(z).float()
        fake_images = G(z)
        D_fake = D(fake_images)
        g_loss = real_loss(D_fake)
        g_loss.backward()
        g_optimizer.step()

        if batch_i % print_every == 0:
            print(
                'Epoch [{:5d}/{:5d}] | d_loss: {:6.4f} | g_loss: {:6.4f}'.format(
                    epoch + 1, num_epochs, d_loss.item(), g_loss.item()
                )
            )

    losses.append((d_loss.item(), g_loss.item()))

    # generate and save sample, fake images
    G.eval()
    samples_z = G(fixed_z)
    samples.append(samples_z)
    G.train()


# Save training generator samples
with open('train_samples.pkl', 'wb') as f:
    pkl.dump(samples, f)

fig, ax = plt.subplots()
losses = np.array(losses)
plt.plot(losses.T[0], label='Discriminator')
plt.plot(losses.T[1], label='Generator')
plt.title("Training Losses")
plt.legend()


def view_samples(epoch, samples):
    fig, axes = plt.subplots(figsize=(7, 7), nrows=4, ncols=4, sharey=True, sharex=True)
    for ax, img in zip(axes.flatten(), samples[epoch]):
        img = img.detach()
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        im = ax.imshow(img.reshape((28, 28)), cmap='Greys_r')


with open('train_samples.pkl', 'rb') as f:
    samples = pkl.load(f)

# -1 indicates final epoch's samples (the last in the list)
view_samples(-1, samples)

rows = 10  # split epochs into 10, so 100/10 = every 10 epochs
cols = 6
fig, axes = plt.subplots(
    figsize=(7, 12), nrows=rows, ncols=cols, sharex=True, sharey=True
)

for sample, ax_row in zip(samples[:: int(len(samples) / rows)], axes):
    for img, ax in zip(sample[:: int(len(sample) / cols)], ax_row):
        img = img.detach()
        ax.imshow(img.reshape((28, 28)), cmap='Greys_r')
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)

# randomly generated, new latent vectors
sample_size = 16
rand_z = np.random.uniform(-1, 1, size=(sample_size, latent_vector_size))
rand_z = torch.from_numpy(rand_z).float()

G.eval()  # eval mode
# generated samples
rand_images = G(rand_z)

# 0 indicates the first set of samples in the passed in list
# and we only have one batch of samples, here
view_samples(0, [rand_images])
