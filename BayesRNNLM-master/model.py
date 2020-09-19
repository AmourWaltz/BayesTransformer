import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
import numpy as np
import math, sys
from bayes_rnn import LSTM, FastBayesLSTM


class RNNLM(nn.Module):   
    def __init__ (self, model, vocsize, embsize, hiddensize, nlayers,
                  no_tied, uncertain=None, position=0):
        super(RNNLM, self).__init__()
        self.model = model.lower()
        self.uncertain = uncertain
        self.encoder = nn.Embedding(vocsize, embsize)
        self.rnns = []
        for l in range(nlayers):
            if model == "lstm":
                if uncertain == 'bayes':
                    self.rnns.append(FastBayesLSTM(hiddensize if l != 0 else embsize,
                        hiddensize if l != nlayers-1 else embsize, position))
                else:
                    self.rnns.append(LSTM(hiddensize if l != 0 else embsize,
                        hiddensize if l != nlayers-1 else embsize))          

        self.rnns = torch.nn.ModuleList(self.rnns)
        self.decoder_bias = nn.Parameter(torch.empty(vocsize))
        if no_tied:
            self.decoder_weight = nn.Parameter(torch.empty(vocsize, embsize))
        else:
            self.decoder_weight = self.encoder.weight       
        self.vocsize = vocsize
        self.embsize = embsize
        self.hiddensize = hiddensize
        self.nlayers = nlayers
        self.no_tied = no_tied
        self.uncertain = uncertain       
        self.init_parameters()

    def init_parameters(self):
        initrange = 1./np.sqrt(self.embsize)
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder_bias.data.uniform_(-initrange, initrange)
        if self.no_tied:
            self.decoder_weight.data.uniform_(-initrange, initrange)
    
    def embed(self, words, dropout=0, scale=None):
        masked_embed_weight = self.encoder.weight
        if dropout:
            mask = self.encoder.weight.data.new().resize_((self.encoder.weight.size(0), 1))\
                .bernoulli_(1-dropout).expand_as(self.encoder.weight)/(1-dropout)
            masked_embed_weight = mask*masked_embed_weight
        padding_idx = self.encoder.padding_idx
        if padding_idx is None:
            padding_idx = -1
        return F.embedding(words, masked_embed_weight,
            padding_idx, self.encoder.max_norm, self.encoder.norm_type,
            self.encoder.scale_grad_by_freq, self.encoder.sparse
        )
    
    def forward(self, inputs, hidden, sents_len):
        new_hidden = []
        for l, rnn in enumerate(self.rnns):
            if('lstm' == self.model):
                new_hidden.append((hidden[l][0].repeat(1, inputs.size(1), 1),
                                   hidden[l][1].repeat(1, inputs.size(1), 1)))
        raw_output = self.embed(inputs)
        if self.model=="lstm":
            for l, rnn in enumerate(self.rnns):
                raw_output, _ = rnn(raw_output, new_hidden[l])

        model_output = F.linear(raw_output, self.decoder_weight, self.decoder_bias)
        model_output = pack_padded_sequence(model_output, sents_len)[0]
        return model_output, raw_output

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return [(weight.new(1, 1, self.hiddensize if l != self.nlayers-1 else self.embsize).zero_(),
                 weight.new(1, 1, self.hiddensize if l != self.nlayers-1 else self.embsize).zero_())
                for l in range(self.nlayers)]
