import os
import torch
import unicodedata
import torch.utils.data as data

from collections import Counter
SOS = '<s>'
EOS = '</s>'
UNK = '<unk>'
NUM = '<num>'


class Vocabulary(object):
    """
    Vocabulary: establish indexes to words.
    """

    def __init__(self, vocfile, use_num=True, data_version=None):
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
            if data_version == 2 or 3:
                num_word = len(str(loop_i)) + 1
                word = word[:-num_word]
                pass
            pass

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
    """
    Establish dictionary: map words into index and complete index sequences.
    Split sentences in train valid and test dataset into batches and convert them to index sequences.
    """

    def __init__(self, path, data_version, valid_batch=10, test_batch=1, use_voc=True):
        self.use_voc = use_voc
        self.dictionary = Dictionary()
        if data_version == 0:
            self.voc = Vocabulary(os.path.join(path, 'voc.txt'), use_num=False, data_version=data_version)
            self.train_data = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid_data = self.tokenize(os.path.join(path, 'valid.txt'))
            self.test_data = self.tokenize(os.path.join(path, 'test.txt'))
            self.valid_loader = None
            self.test_loader = None
            pass
        elif data_version == 1:
            self.voc = Vocabulary(os.path.join(path, 'voc.txt'), use_num=False, data_version=data_version)
            self.train_data = self.tokenize(os.path.join(path, 'train.txt'))
            self.valid_data = TextDataset(os.path.join(path, 'valid.txt'), self.voc)
            self.test_data = TextDataset(os.path.join(path, 'test.txt'), self.voc)
            # valid_lens = list(map(len, self.valid_data.words))
            # test_lens = list(map(len, self.test_data.words))
            # print("val, test:", valid_lens, test_lens)
            self.valid_loader = data.DataLoader(self.valid_data, batch_size=valid_batch,
                                                shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
            self.test_loader = data.DataLoader(self.test_data, batch_size=test_batch,
                                               shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
            pass
        elif data_version == 2:
            self.voc = Vocabulary(os.path.join(path, 'words.txt'), use_num=False, data_version=data_version)
            self.train_data = self.tokenize(os.path.join(path, 'fisher.txt'))
            self.valid_data = TextDataset(os.path.join(path, 'dev.txt'), self.voc)
            self.test_data = TextDataset(os.path.join(path, 'swbd.txt'), self.voc)
            self.valid_loader = data.DataLoader(self.valid_data, batch_size=valid_batch,
                                                shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
            self.test_loader = data.DataLoader(self.test_data, batch_size=test_batch,
                                               shuffle=False, num_workers=0, collate_fn=collate_fn, drop_last=False)
            pass
        elif data_version == 3:
            self.voc = Vocabulary(os.path.join(path, 'words.txt'), use_num=False, data_version=data_version)
            self.train_data = self.tokenize(os.path.join(path, 'fisher.txt'))
            self.valid_data = self.tokenize(os.path.join(path, 'dev.txt'))
            self.test_data = self.tokenize(os.path.join(path, 'swbd.txt'))
            self.valid_loader = None
            self.test_loader = None
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


# Using Dataset to load txt data
class TextDataset(data.Dataset):
    def __init__(self, txtfile, voc):
        """
        :param txtfile: Path of txt.
        :param voc: Established vocabulary.
        """
        self.words, self.ids = self.token(txtfile, voc)
        self.nline = len(self.ids)
        self.n_sents = len(self.ids)
        self.n_words = sum([len(ids) for ids in self.ids])
        self.n_unks = len([index for ids in self.ids for index in ids if index == voc.word2id(UNK)])

    @staticmethod
    # Devide passage into words.
    def token(txtfile, voc):
        assert os.path.exists(txtfile)
        lines = open(txtfile, 'r').readlines()
        words, ids = [], []
        for _, line in enumerate(lines):
            tokens = line.strip().split()
            # print("current sentence: ", tokens)
            if len(tokens) == 0:
                continue
                pass
            pass

            # Convert word sequence into index sequence.
            # Words append a list of a sentence.
            words.append([SOS])
            # print(words)
            ids.append([voc.word2id(SOS)])
            for token in tokens:
                # print("Voc lengths: ", len(voc.idx2word), voc.word2id('wwwwww'))
                if voc.word2id(token) + 1 < len(voc.idx2word):
                    words[-1].append(token)
                    ids[-1].append(voc.word2id(token))
                    pass
                else:
                    words[-1].append(UNK)
                    ids[-1].append(voc.word2id(UNK))
                    # Test on 7.5.
                    # print("word UNK", words[-1])
                    pass
                pass
            pass

            # Ends to each sentence.
            words[-1].append(EOS)
            ids[-1].append(voc.word2id(EOS))
            pass
        pass

        return words, ids

    def __len__(self):
        return self.n_sents

    def __repr__(self):
        return '#Sents=%d, #Words=%d, #UNKs=%d' % (self.n_sents, self.n_words, self.n_unks)

    def __getitem__(self, index):
        return self.ids[index]


# Merge a list of samples with variable sizes by padding to form a mini-batch of Tensors.
def collate_fn(batch):
    # map(func, list): map list using func.
    # sent_lens = torch.LongTensor(list(map(len, batch)))
    sent_len_list = torch.tensor(list(map(len, batch))).long()
    max_len = sent_len_list.max().numpy()
    batchsize = len(batch)
    # Build a variable identical with sent_batch using new_zeros,
    # global max_length
    sent_batch = sent_len_list.new_zeros((batchsize, max_len))
    # zip(): Pack objects into tuples.
    # Align sentences by padding 0.
    # global max_length
    for idx, (sent, sent_len) in enumerate(zip(batch, sent_len_list)):
        sent_batch[idx, :sent_len] = torch.tensor(sent).long()
        pass
    pass

    # Resort batch by descending sentences length list.
    sent_length, perm_idx = sent_len_list.sort(0, descending=True)
    sent_batch = sent_batch[perm_idx]
    sent_batch = sent_batch.t().contiguous()
    # After transpose and contiguous, operate on dim=0.
    inputs_batch = sent_batch[0: max_len - 1]
    targets_batch = sent_batch[1: max_len]
    # sent_length.sub_(1) # test on 7.5.
    sent_length -= 1
    pass

    return inputs_batch, targets_batch, sent_length
