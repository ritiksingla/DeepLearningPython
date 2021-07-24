import re
from collections import Counter
import numpy as np
import random
import torch
from torch import nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Download and load the data
with open('/content/sample_data/text8') as f:
    text = f.read()


def preprocess(text):

    # Replace punctuation with tokens so we can use them in our model
    text = text.lower()
    text = text.replace('.', ' <PERIOD> ')
    text = text.replace(',', ' <COMMA> ')
    text = text.replace('"', ' <QUOTATION_MARK> ')
    text = text.replace(';', ' <SEMICOLON> ')
    text = text.replace('!', ' <EXCLAMATION_MARK> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    text = text.replace('(', ' <LEFT_PAREN> ')
    text = text.replace(')', ' <RIGHT_PAREN> ')
    text = text.replace('--', ' <HYPHENS> ')
    text = text.replace('?', ' <QUESTION_MARK> ')
    # text = text.replace('\n', ' <NEW_LINE> ')
    text = text.replace(':', ' <COLON> ')
    words = text.split()

    # Remove all words with  5 or fewer occurences
    word_counts = Counter(words)
    trimmed_words = [word for word in words if word_counts[word] > 5]

    return trimmed_words


def create_lookup_tables(words):
    word_counts = Counter(words)
    # sorting the words from most to least frequent in text occurrence
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    # create int_to_vocab dictionaries
    int_to_vocab = {ii: word for ii, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: ii for ii, word in int_to_vocab.items()}

    return vocab_to_int, int_to_vocab


# Tokenizing the text
words = preprocess(text)
vocab_to_int, int_to_vocab = create_lookup_tables(words)
int_words = [vocab_to_int[word] for word in words]

# Subsampling int_words
"""
- Subsampling technique refers to removing words that have high frequency like as, of, the, etc.

- Probability of discarding the word
Prob(w_i) = 1 - sqrt(threshold/frequency(w_i))
- Frequency of word w_i is defined as count(w_i)/total_count
"""

threshold = 1e-5
word_counts = Counter(int_words)
total_count = len(word_counts)
p_drop = {
    word: 1 - np.sqrt(threshold * total_count / cnt)
    for word, cnt in word_counts.items()
}
train_words = [word for word in int_words if random.random() < (1 - p_drop[word])]

# Making Batches
def get_target(words, idx, max_window_size=5):
    window_size = np.random.randint(low=1, high=max_window_size + 1)
    start = idx - window_size if idx - window_size > 0 else 0
    finish = idx + window_size
    target_words = words[start:idx]
    target_words = np.append(target_words, words[idx + 1 : finish + 1])
    return list(target_words)


# Generating Batches
def get_batches(words, batch_size, window_size=5):
    n_batches = len(words) // batch_size
    for batch in range(n_batches):
        x = []
        y = []
        cur_batch = words[batch * batch_size : (batch + 1) * batch_size]
        for word_of_interest in range(batch_size):
            batch_x = cur_batch[word_of_interest]
            batch_y = get_target(
                words=cur_batch, idx=word_of_interest, max_window_size=window_size
            )
            y.extend(batch_y)
            x.extend([batch_x] * len(batch_y))
        yield x, y


# Cosine Similarity between word vectors
def cosine_similarity(embedding, valid_size=16, valid_window=100, device='cpu'):
    embed_vectors = embedding.weight
    magnitudes = embed_vectors.pow(2).sum(dim=1).sqrt().unsqueeze(0)

    valid_examples = np.array(random.sample(range(valid_window), valid_size // 2))
    valid_examples = np.append(
        valid_examples, random.sample(range(1000, 1000 + valid_window), valid_size // 2)
    )
    valid_examples = torch.LongTensor(valid_examples).to(device)
    valid_vectors = embedding(valid_examples)
    similarities = torch.mm(valid_vectors, embed_vectors.t()) / magnitudes
    return valid_examples, similarities


# Skip Gram Model
class SkipGram(nn.Module):
    def __init__(self, n_vocab, n_embed):
        super(SkipGram, self).__init__()
        self.embed = nn.Embedding(num_embeddings=n_vocab, embedding_dim=n_embed)
        self.output = nn.Linear(n_embed, n_vocab)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.embed(x)
        log_ps = self.log_softmax(self.output(x))
        return log_ps


# Training the model
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_dim = 300

# As we don't know which words from vocab got filtered, we have to use whole dict for input and output
model = SkipGram(len(vocab_to_int), embedding_dim).to(device)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 500
steps = 0
epochs = 5

for epoch in range(epochs):
    for inputs, targets in get_batches(train_words, batch_size=512, window_size=5):
        steps += 1
        inputs, targets = torch.LongTensor(inputs).to(device), torch.LongTensor(
            targets
        ).to(device)

        log_ps = model(inputs)
        optimizer.zero_grad()
        loss = criterion(log_ps, targets)
        loss.backward()
        optimizer.step()

        if steps % print_every == 0:
            print(f"Epoch : {epoch} | Step : {steps}")
            valid_ex, valid_sim = cosine_similarity(model.embed, device=device)
            _, closest_idxs = valid_sim.topk(6)
            valid_ex, closest_idxs = valid_ex.to("cpu"), closest_idxs.to("cpu")
            for ii, valid_idx in enumerate(valid_ex):
                closest_words = [
                    int_to_vocab[idx.item()] for idx in closest_idxs[ii][1:]
                ]
                print(f"{int_to_vocab[valid_idx.item()]} -> {', '.join(closest_words)}")

# Visualizing the word vectors
embeddings = model.embed.weight.to("cpu").data.numpy()
viz_words = 600
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize=(16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx, :], color="steelblue")
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

# Negative Sampling

class SkipGramNeg(nn.Module):
    def __init__(self, n_vocab, n_embed, noise_dist=None):
        super().__init__()
        
        self.n_vocab = n_vocab
        self.n_embed = n_embed
        self.noise_dist = noise_dist
        
        # define embedding layers for input and output words
        self.in_embed = nn.Embedding(n_vocab, n_embed)
        self.out_embed = nn.Embedding(n_vocab, n_embed)
        
        # Initialize embedding tables with uniform distribution
        # I believe this helps with convergence
        self.in_embed.weight.data.uniform_(-1, 1)
        self.out_embed.weight.data.uniform_(-1, 1)
        
    def forward_input(self, input_words):
        input_vectors = self.in_embed(input_words)
        return input_vectors
    
    def forward_output(self, output_words):
        output_vectors = self.out_embed(output_words)
        return output_vectors
    
    def forward_noise(self, batch_size, n_samples):
        """ Generate noise vectors with shape (batch_size, n_samples, n_embed)"""
        if self.noise_dist is None:
            # Sample words uniformly
            noise_dist = torch.ones(self.n_vocab)
        else:
            noise_dist = self.noise_dist
            
        # Sample words from our noise distribution
        noise_words = torch.multinomial(noise_dist,
                                        batch_size * n_samples,
                                        replacement=True)
        
        device = "cuda" if model.out_embed.weight.is_cuda else "cpu"
        noise_words = noise_words.to(device)
        
        noise_vectors = self.out_embed(noise_words).view(batch_size, n_samples, self.n_embed)
        
        return noise_vectors

class NegativeSamplingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input_vectors, output_vectors, noise_vectors):
        
        batch_size, embed_size = input_vectors.shape
        
        # Input vectors should be a batch of column vectors
        input_vectors = input_vectors.view(batch_size, embed_size, 1)
        
        # Output vectors should be a batch of row vectors
        output_vectors = output_vectors.view(batch_size, 1, embed_size)
        
        # bmm = batch matrix multiplication
        # correct log-sigmoid loss
        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log()
        out_loss = out_loss.squeeze()
        
        # incorrect log-sigmoid loss
        noise_loss = torch.bmm(noise_vectors.neg(), input_vectors).sigmoid().log()
        noise_loss = noise_loss.squeeze().sum(1)  # sum the losses over the sample of noise vectors

        # negate and sum correct and noisy log-sigmoid losses
        # return average batch loss
        return -(out_loss + noise_loss).mean()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get our noise distribution
# Using word frequencies calculated earlier in the notebook
word_freqs = np.array(sorted(freqs.values(), reverse=True))
unigram_dist = word_freqs/word_freqs.sum()
noise_dist = torch.from_numpy(unigram_dist**(0.75)/np.sum(unigram_dist**(0.75)))

# instantiating the model
embedding_dim = 300
model = SkipGramNeg(len(vocab_to_int), embedding_dim, noise_dist=noise_dist).to(device)

# using the loss that we defined
criterion = NegativeSamplingLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.003)

print_every = 1500
steps = 0
epochs = 5

# train for some number of epochs
for e in range(epochs):
    
    # get our input, target batches
    for input_words, target_words in get_batches(train_words, 512):
        steps += 1
        inputs, targets = torch.LongTensor(input_words), torch.LongTensor(target_words)
        inputs, targets = inputs.to(device), targets.to(device)
        
        # input, output, and noise vectors
        input_vectors = model.forward_input(inputs)
        output_vectors = model.forward_output(targets)
        noise_vectors = model.forward_noise(inputs.shape[0], 5)

        # negative sampling loss
        loss = criterion(input_vectors, output_vectors, noise_vectors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # loss stats
        if steps % print_every == 0:
            print("Epoch: {}/{}".format(e+1, epochs))
            print("Loss: ", loss.item()) # avg batch loss at this point in training
            valid_examples, valid_similarities = cosine_similarity(model.in_embed, device=device)
            _, closest_idxs = valid_similarities.topk(6)

            valid_examples, closest_idxs = valid_examples.to('cpu'), closest_idxs.to('cpu')
            for ii, valid_idx in enumerate(valid_examples):
                closest_words = [int_to_vocab[idx.item()] for idx in closest_idxs[ii]][1:]
                print(int_to_vocab[valid_idx.item()] + " | " + ', '.join(closest_words))
            print("...\n")

%matplotlib inline
%config InlineBackend.figure_format = 'retina'
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
embeddings = model.in_embed.weight.to("cpu").data.numpy()
viz_words = 380
tsne = TSNE()
embed_tsne = tsne.fit_transform(embeddings[:viz_words, :])

fig, ax = plt.subplots(figsize = (16, 16))
for idx in range(viz_words):
    plt.scatter(*embed_tsne[idx,:], color = "steelblue")
    plt.annotate(int_to_vocab[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha = 0.7)