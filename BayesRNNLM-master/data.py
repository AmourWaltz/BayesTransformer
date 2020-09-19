import os
import torch
import torch.utils.data as Data
import numpy as np
import unicodedata

from utils import device

SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
NUM = '<num>'

class Vocabulary(object):
    
    def __init__ (self, vocfile, use_num=True):
        self.use_num = use_num
        self.word2idx = {}
        self.idx2word = {}
        self.word2idx[SOS] = 0
        self.idx2word[0] = SOS
        self.word2idx[EOS] = 1
        self.idx2word[1] = EOS
        words = open(vocfile, 'r').read().strip().split('\n')
        for word in words:
            if self.use_num and self.is_number(word):
                word = NUM
            if word not in self.word2idx and word!=UNK:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
        self.word2idx[UNK] = len(self.word2idx)
        self.idx2word[len(self.word2idx)] = UNK
        self.vocsize = len(self.word2idx)

    def word2id (self, word):
        if self.use_num and self.is_number(word):
            word = NUM
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx[UNK]

    def id2word(self, idx):
        if idx in self.idx2word:
            return self.idx2word[idx]
        else:
            return UNK

    def is_number(self, word):
        word = word.replace(',', '')   # 10,000 -> 10000
        word = word.replace(':', '')   # 5:30 -> 530
        word = word.replace('-', '')   # 17-08 -> 1708
        word = word.replace('/', '')   # 17/08/1992 -> 17081992
        word = word.replace('th', '')  # 20th -> 20
        word = word.replace('rd', '')  # 93rd -> 20
        word = word.replace('nd', '')  # 22nd -> 20
        word = word.replace('m', '')   # 20m -> 20
        word = word.replace('s', '')   # 20s -> 20
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

# using Dataset to load txt data
class TextDataset(Data.Dataset):
    
    def __init__(self, txtfile, voc):
        self.words, self.ids = self.tokenize(txtfile, voc)
        self.nline = len(self.ids)
        self.n_sents = len(self.ids)
        self.n_words = sum([len(ids) for ids in self.ids])
        self.n_unks = len([id for ids in self.ids for id in ids if id == voc.word2id(UNK)])

    def tokenize(self, txtfile, voc):
        assert os.path.exists(txtfile)
        lines = open(txtfile, 'r').readlines()
        words, ids = [], []
        for i, line in enumerate(lines):
            tokens = line.strip().split()
            if(len(tokens) == 0):
                continue
            words.append([SOS])
            ids.append([voc.word2id(SOS)])
            for token in tokens:
                if voc.word2id(token) < len(voc.idx2word):
                    words[-1].append(token)
                    ids[-1].append(voc.word2id(token))
                else:
                    words[-1].append(UNK)
                    ids[-1].append(voc.word2id(UNK))
            words[-1].append(EOS)
            ids[-1].append(voc.word2id(EOS))
        return words, ids

    def __len__(self):
        return self.n_sents

    def __repr__(self):
        return '#Sents=%d, #Words=%d, #UNKs=%d'%(self.n_sents, self.n_words, self.n_unks)
    
    def __getitem__ (self, index):
        return self.ids[index]

class Corpus(object):
    
    def __init__(self, data_dir, train_batch_size, valid_batch_size, test_batch_size):
        self.voc = Vocabulary(os.path.join(data_dir, 'voc.txt'))
        self.train_data = TextDataset(os.path.join(data_dir, 'train.txt'), self.voc)
        self.valid_data = TextDataset(os.path.join(data_dir, 'valid.txt'), self.voc)
        self.test_data = TextDataset(os.path.join(data_dir, 'test.txt'), self.voc)
        self.train_loader = Data.DataLoader(self.train_data, batch_size=train_batch_size,
            shuffle=True, num_workers=0, collate_fn=collate_fn, drop_last=False)
        self.valid_loader = Data.DataLoader(self.valid_data, batch_size=valid_batch_size,
            shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
        self.test_loader = Data.DataLoader(self.test_data, batch_size=test_batch_size,
            shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)

    def __repr__(self):
        return 'Train: %s\n'%(self.train_data)+\
            'Valid: %s\n'%(self.valid_data)+\
            'Test: %s\n'%(self.test_data)

def collate_fn(batch):
    sent_lens = torch.LongTensor(list(map(len, batch)))
    max_len = sent_lens.max()
    batchsize = len(batch)
    sent_batch = sent_lens.new_zeros(batchsize, max_len)
    for idx, (sent, sent_len) in enumerate(zip(batch, sent_lens)):
        sent_batch[idx, :sent_len] = torch.LongTensor(sent)
    sent_lens, perm_idx = sent_lens.sort(0, descending=True)
    sent_batch = sent_batch[perm_idx]
    sent_batch = sent_batch.t().contiguous()
    inputs = sent_batch[0:max_len-1]
    targets = sent_batch[1:max_len]
    sent_lens.sub_(1)
    return inputs.to(device), targets.to(device), sent_lens.to(device)


if __name__ == '__main__':
    corpus = Corpus('data/ami', 8, 16, 1)
    print(len(corpus.voc))
    print(corpus.voc.id2word(len(corpus.voc)), corpus.voc.word2id('<unk>'),corpus.voc.word2id('<s>'), corpus.voc.word2id('</s>'))
    for i, (inputs, targets, sent_lens) in enumerate(corpus.train_loader):
        print(i, inputs, targets, sent_lens)
        break
    for i, (inputs, targets, sent_lens) in enumerate(corpus.valid_loader):
        print(i, inputs.size(), targets.size(), sent_lens)
        break
    for i, (inputs, targets, sent_lens) in enumerate(corpus.test_loader):
        print(i, inputs.size(), targets.size(), sent_lens)
        break
