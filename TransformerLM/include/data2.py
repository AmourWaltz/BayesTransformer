import os
import torch
from data import data_version
from data import Vocabulary
from collections import Counter


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []
        self.counter = Counter()
        self.total = 0

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        token_id = self.word2idx[word]
        self.counter[token_id] += 1
        self.total += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path, use_voc=False):
        self.use_voc = use_voc
        self.dictionary = Dictionary()
        if data_version == 1:
            self.train = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test = self.tokenize(os.path.join(path, 'test.txt'))
            pass
        elif data_version == 2:
            self.voc = Vocabulary(os.path.join(path, 'words.txt'), use_num=False)
            self.train = self.tokenize(os.path.join(path, 'fisher.txt'))
            self.valid = self.tokenize(os.path.join(path, 'dev.txt'))
            self.test = self.tokenize(os.path.join(path, 'swbd.txt'))
            pass
        pass

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['</s>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # print(self.voc.word2idx)
        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            # print(ids)
            token = 0
            for line in f:
                words = line.split() + ['</s>']
                for word in words:
                    # print(word)
                    if self.use_voc is True:
                        ids[token] = self.voc.word2id(word)
                        pass
                    else:
                        ids[token] = self.dictionary.word2idx[word]
                        pass
                    pass
                    token += 1

        return ids
