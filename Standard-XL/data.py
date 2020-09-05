import os
import torch
import unicodedata

from collections import Counter
SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
NUM = '<num>'


class Vocabulary(object):
    """
    Vocabulary: establish indexes to words.
    """

    def __init__(self, vocfile, use_num=True):
        super(Vocabulary, self).__init__()
        self.use_num = use_num
        self.word2idx = {}
        self.word_feq = {}
        self.idx2word = dict()
        self.word2idx[SOS] = 0
        self.idx2word[0] = SOS
        self.word2idx[EOS] = 1
        self.idx2word[1] = EOS
        words = open(vocfile, 'r').read().strip().split('\n')
        # Look up the vocabulary as a list.
        # print("vocabulary: ", words)

        for loop_i, word in enumerate(words):
            num_word = len(str(loop_i)) + 1
            word = word[:-num_word]
            # print(word)
            if self.use_num and self.is_number(word):
                word = NUM
                pass
            if word not in self.word2idx and word != UNK:
                # Establish the dictionary.
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                pass
            pass
        pass

        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word[len(self.word2idx)] = UNK
        self.vocsize = len(self.word2idx)
        pass

    # Convert word to its index.
    def word2id(self, word):
        if self.use_num and self.is_number(word):
            word = NUM
            pass
        pass

        if word in self.word2idx:
            # self.word_feq[word] += 1
            return self.word2idx[word]
        else:
            # if word == 'email':
            #     self.email += 1
            #     pass
            # elif word == 'website':
            #     self.website += 1
            #     pass
            # elif word == '-em':
            #     self.em += 1
            #     pass
            # pass
            return self.word2idx[UNK]
        pass

    # Convert index to word.
    def id2word(self, idx):
        if idx in self.idx2word:
            return self.idx2word[idx]
        else:
            return UNK

    @staticmethod
    def is_number(word):
        word = word.replace(',', '')  # 10,000 -> 10000
        word = word.replace(':', '')  # 5:30 -> 530
        word = word.replace('-', '')  # 17-08 -> 1708
        word = word.replace('/', '')  # 17/08/1992 -> 17081992
        word = word.replace('th', '')  # 20th -> 20
        word = word.replace('rd', '')  # 93rd -> 20
        word = word.replace('nd', '')  # 22nd -> 20
        word = word.replace('m', '')  # 20m -> 20
        word = word.replace('s', '')  # 20s -> 20
        try:
            float(word)
            return True
        except ValueError:
            pass
        try:
            unicodedata.numeric(word)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def __len__(self):
        return self.vocsize

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
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.voc = Vocabulary(os.path.join(path, 'words.txt'), use_num=False)
        self.train = self.tokenize(os.path.join(path, 'fisher.txt'))
        self.valid = self.tokenize(os.path.join(path, 'dev.txt'))
        self.test = self.tokenize(os.path.join(path, 'swbd.txt'))

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

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            token = 0
            for line in f:
                words = line.split() + ['</s>']
                for word in words:
                    ids[token] = self.voc.word2id(word)
                    token += 1

        return ids
