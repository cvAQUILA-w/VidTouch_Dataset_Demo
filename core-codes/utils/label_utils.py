# utils/label_utils.py

import torch

class LabelEncoder:
    def __init__(self, class_list):
        self.class2idx = {c: i for i, c in enumerate(class_list)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}

    def encode(self, label):
        return self.class2idx[label]

    def decode(self, idx):
        return self.idx2class[idx]

class MultiHotEncoder:
    def __init__(self, vocab):
        self.vocab = vocab
        self.word2idx = {w: i for i, w in enumerate(vocab)}

    def encode(self, words):
        vec = torch.zeros(len(self.vocab))
        for w in words:
            if w in self.word2idx:
                vec[self.word2idx[w]] = 1.0
        return vec

    def decode(self, vec, threshold=0.5):
        return [self.vocab[i] for i, v in enumerate(vec) if v > threshold]