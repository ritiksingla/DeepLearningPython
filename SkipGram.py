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
