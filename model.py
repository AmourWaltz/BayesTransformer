from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

#from locked_dropout import LockedDropout


class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5,
                 tie_weights=False):
        super(RNNModel, self).__init__()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError("""An invalid option for `--model` was supplied,
                      options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity,
                              dropout=dropout)
        self.decoder = nn.Linear(nhid, ntoken)

        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, x, hidden):
        emb = self.drop(self.encoder(x))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        return weight.new_zeros(self.nlayers, bsz, self.nhid)


class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the
        tokens in the sequence. The positional encodings have the same dimension
        as the embeddings, so that the two can be summed. Here, we use sine and
        cosine functions of different frequencies.
    .. math::
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """

        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., bias=True, add_bias_kv=False, add_zero_attn=False, kdim=None, vdim=None):
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # self.in_proj_weight = Parameter(torch.empty(3 * embed_dim, embed_dim))
        self.qkv_net = nn.Linear(embed_dim, 3 * embed_dim)

        self.drop = nn.Dropout(dropout)

        # self.out_proj = _LinearWithBias(embed_dim, embed_dim)
        self.o_net = nn.Linear(embed_dim, embed_dim)

        # self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            nn.init.xavier_uniform_(self.qkv_net.weight)

        #if self.in_proj_bias is not None:
        nn.init.constant_(self.qkv_net.bias, 0.)
        nn.init.constant_(self.o_net.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        scaling = float(self.head_dim) ** -0.5
        tgt_len, bsz, embed_dim = query.size()

        q, k, v = self.qkv_net(query).chunk(3, dim=-1)
        q = q * scaling
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0)
                if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 2D attn_mask is not correct.')
            elif attn_mask.dim() == 3:
                # XBY 9.30: num_heads to self.num_heads
                if list(attn_mask.size()) != [bsz * self.num_heads, query.size(0), key.size(0)]:
                    raise RuntimeError('The size of the 3D attn_mask is not correct.')
            else:
                raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))

        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)

        attn_output_weights = torch.bmm(q, k.transpose(1, 2))
        assert list(attn_output_weights.size()) == [bsz * self.num_heads, tgt_len, src_len] 
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_output_weights.masked_fill_(attn_mask, float('-inf'))
            else:
                attn_output_weights += attn_mask

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = self.drop(attn_output_weights)

        attn_output = torch.bmm(attn_output_weights, v)
        assert list(attn_output.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.o_net(attn_output) 

        if need_weights:
            # average attention weights over heads
            attn_output_weights = attn_output_weights.view(bsz, self.num_heads, tgt_len, src_len)
            return attn_output, attn_output_weights.sum(dim=1) / self.num_heads
        else:
            return attn_output, None

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)

        # XBY 9.30: 0 to align the outputs to BayesTransformer
        return src, 0


class BayesTransformerEncoderLayer(nn.Module):
    """
    XBY 9.30. 
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(BayesTransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.dim_ff = dim_feedforward
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout)
        # Each weight under Gaussian Distribution is determined by two parameters, mean and variance.
        # self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.weight_mean1 = nn.Parameter(torch.rand(self.dim_ff, self.d_model), requires_grad=True)
        self.weight_mean2 = nn.Parameter(torch.rand(self.d_model, self.dim_ff), requires_grad=True)
        self.dropout = nn.Dropout(dropout)       
        # self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.weight_std1 = nn.Parameter(torch.rand(self.dim_ff, self.d_model), requires_grad=True)
        self.weight_std2 = nn.Parameter(torch.rand(self.d_model, self.dim_ff), requires_grad=True)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.GELU()

    def reset_parameters(self):
        stdm = 1. / math.sqrt(self.d_model)
        stdi = 1. / math.sqrt(self.dim_ff)
        self.weight_mean1.data.uniform_(-stdm, stdm)
        self.weight_mean2.data.uniform_(-stdi, stdi)
        # Note that variances are calculated by log.
        self.weight_std1.data.uniform_(2*np.log(stdm), 1*np.log(stdm))
        self.weight_std2.data.uniform_(2*np.log(stdi), 1*np.log(stdi))

    def sample_weight_diff(self):
        if self.training:
            # Sampling process, sample_value = mean + epsilon * variance, where epsilon is sampled from N(0, 1).
            weight_lgstd_1 = torch.exp(self.weight_std1)
            epsilon = weight_lgstd_1.new_zeros(*weight_lgstd_1.size()).normal_()
            weight_diff_1 = epsilon*weight_lgstd_1
            weight_lgstd_2 = torch.exp(self.weight_std2)
            epsilon = weight_lgstd_2.new_zeros(*weight_lgstd_2.size()).normal_()
            weight_diff_2 = epsilon*weight_lgstd_2
            return weight_diff_1, weight_diff_2
        return 0, 0

    def kl_divergence(self):
        kl = 0
        theta_mean = torch.cat([self.weight_mean1, self.weight_mean2.t()], -1)
        theta_std = torch.cat([self.weight_std1, self.weight_std2.t()], -1)
        kl += torch.mean(theta_mean ** 2. - theta_std * 2. + torch.exp(theta_std * 2)) / 2.
        # kl += torch.mean(theta_mean ** 2. + theta_std ** 2 + 2 * torch.log(theta_std)) / 2.
        return kl

    def forward(self, src, src_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        # Using Gaussian sampling to add the variance.
        # src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        weight1 = self.weight_mean1*1.
        weight2 = self.weight_mean2*1.
        weight1_diff, weight2_diff = self.sample_weight_diff()
        weight1 += weight1_diff
        weight2 += weight2_diff

        src1 = F.linear(src, weight1)
        src2 = F.linear(self.activation(src1), weight2)
        ff_kl = self.kl_divergence()

        src = src + self.dropout2(src2)
        src = self.norm2(src)

        return src, ff_kl


class TransformerModel(nn.Module):
    """Container module with an encoder, a transformer module, and a decoder."""

    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(TransformerModel, self).__init__()
        # try:
        #     from torch.nn import TransformerEncoder, TransformerEncoderLayer
        # except ImportError:
        #     raise ImportError('TransformerEncoder module does not exist in '
        #                       'PyTorch 1.1 or lower.')
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        self.transformerlayers = nn.ModuleList()

        # XBY 9.30: Change the first Transformer layer followed by embedding to Bayesian FFN.
        self.transformerlayers.append(BayesTransformerEncoderLayer(ninp, nhead, nhid, dropout))

        for i in range(nlayers - 1):
            self.transformerlayers.append(
                    TransformerEncoderLayer(ninp, nhead, nhid, dropout)
                )
        # encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout,
        #                                          activation)
        # self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal '
                                 'to emsize.')
            self.decoder.weight = self.encoder.weight
        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
                mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.encoder.weight, -initrange, initrange)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -initrange, initrange)

    def forward(self, src, has_mask=True):
        if has_mask:
            device = src.device
            if self.src_mask is None or self.src_mask.size(0) != len(src):
                mask = self._generate_square_subsequent_mask(len(src)).to(device)
                self.src_mask = mask
        else:
            self.src_mask = None
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        # output = self.transformerlayers(src, self.src_mask)
        output = src 

        # XBY 9.30: kl_loss is used to calculate total kl_loss
        kl_loss = 0
        for mod in self.transformerlayers:
            output, kl = mod(output, src_mask=self.src_mask)
            kl_loss += kl

        output = self.decoder(output)
        return output
