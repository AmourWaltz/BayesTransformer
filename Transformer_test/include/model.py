import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as func


# Scaled dot-product attention mechanism.
class ScaledDotProductAttention(nn.Module):
    """
    An attention function can be described as a query and a set of key-value pairs to an output,
    where the query, keys, values, and output are all vectors.
    The output is computed as a weighted sum of the values,
    where the weight assigned to each value is computed
    by a compatibility of the query with the corresponding key.
    """
    def __init__(self, attention_dropout=0.0):
        super(ScaledDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(attention_dropout)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, query, key, value, scale=None, attn_mask=None):
        """
        :param query: shape [B, L_q, D_q]
        :param key: shape [B, L_k, D_k]
        :param value: shape [B, L_v, D_v]
        :param scale: make Softmax in low gradient
        :param attn_mask: masking tensor，shape [B, L_q, L_k]
        :return: context tensor as z and attetention tensor
        """
        attention = torch.bmm(query, key.transpose(1, 2))
        if scale:
            attention = attention * scale
            pass
        pass

        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask, -np.inf)
            pass
        pass

        attention = self.softmax(attention)
        attention = self.dropout(attention)
        context = torch.bmm(attention, value)

        return context, attention


# Multi-head attention architecture.
class MultiHeadAttention(nn.Module):
    """
    We found it beneﬁcial to linearly project the queries, keys and values h times with different,
    learned linear projections to d_k , d_k and d_v dimensions, respectively.
    We then perform the attention function in parallel, yielding d_v-dimensional output values.
    These are concatenated and once again projected, resulting in the ﬁnal values,
    """
    def __init__(self, model_dim=512, num_heads=8, dropout=0.0):
        super(MultiHeadAttention, self).__init__()
        self.dim_per_head = model_dim // num_heads
        self.num_heads = num_heads
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, key, value, query, attn_mask=None):
        """
        :param key: shape [B, L_k, D_k]
        :param value: shape [B, L_v, D_v]
        :param query: shape [B, L_q, D_q]
        :param attn_mask: masking tensor，shape [B, L_q, L_k]
        :return:
        """
        residual = query

        dim_per_head = self.dim_per_head
        num_heads = self.num_heads
        batch_size = key.size(0)

        # linear projection
        key = self.linear_k(key)
        value = self.linear_v(value)
        query = self.linear_q(query)

        # split by heads, k q v shape [B * 8, L, 64]
        key = key.view(batch_size * num_heads, -1, dim_per_head)
        value = value.view(batch_size * num_heads, -1, dim_per_head)
        query = query.view(batch_size * num_heads, -1, dim_per_head)

        # print(attn_mask)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(num_heads, 1, 1)
            pass
        pass

        # scaled dot-product attention
        scale = (key.size(-1) // num_heads) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        # concat heads
        context = context.view(batch_size, -1, dim_per_head * num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


# To align sequences with different lengths.
def padding_mask(seq_k, seq_q):
    """
    :param seq_k: shape [B, L]
    :param seq_q: shape [B, L]
    :return: pad_mask shape [B, L, L]
    """
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)

    return pad_mask


# To make latter word invisible for present decoder.
def sequence_mask(seq):
    """
    :param seq: shape [B, L]
    :return: mask shape [B, L, L]
    """
    batch_size, seq_len = seq.size()
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    return mask


# Compute the positional encodings once in trigonometric space.
class PositionalEncoding(nn.Module):
    """
    Where pos is the position and i is the dimension.
    That is, each dimension of the positional encoding corresponds to a sinusoid.
    The wavelengths form a geometric progression from 2π to 10000 · 2π.
    We chose this function because we hypothesized
    it would allow the model to easily learn to attend by relative positions,
    since for any ﬁxed offset k, PE pos+k can be represented as a linear function of PE pos .
    """
    def __init__(self, d_model, max_seq_len):
        """
        :param d_model: sclar, model dimension: 512 in Attention Is All You Need
        :param max_seq_len: sclar, the maximum of input sequence
        """
        super(PositionalEncoding, self).__init__()

        # Position Embedding matrix
        position_encoding = np.array([
            [pos / np.power(10000, 2.0 * (j // 2) / d_model) for j in range(d_model)]
            for pos in range(max_seq_len)])
        # sin for even points，cos for odd points
        position_encoding[:, 0::2] = np.sin(position_encoding[:, 0::2])
        position_encoding[:, 1::2] = np.cos(position_encoding[:, 1::2])

        # add a zero vector in 1st line of PE for positional encoding of PAD
        pad_row = torch.zeros([1, d_model])
        position_encoding = torch.from_numpy(position_encoding).float()
        position_encoding = torch.cat((pad_row, position_encoding))
        self.position_encoding = nn.Embedding(max_seq_len + 1, d_model)
        self.position_encoding.weight = nn.Parameter(position_encoding, requires_grad=False)

    def forward(self, input_len):
        """
        :param input_len: shape [B, 1], represent each length of sequence
        :return: position embeddings of the sequence with alignment
        """
        # find out the max_length of sequences
        # print(input_len)
        # for length in input_len:
        #     print(length)
        input_len = input_len.clone().detach()
        max_len = torch.max(input_len)
        # add 0 behind original sequences to align and avoid 1st line of PAD(0)

        # print(length, max_len)
        if torch.cuda.is_available():
            input_pos = torch.tensor(
                [list(range(1, length.cpu().numpy() + 1)) + [0] * (max_len.cpu().numpy() - length.cpu().numpy())
                 for length in input_len]).long()
            input_pos = input_pos.cuda()
            pass
        else:
            input_pos = torch.tensor([list(range(1, length.numpy() + 1)) + [0] * (max_len.numpy() - length.numpy())
                                      for length in input_len]).long()
            pass
        pass
        # print(input_pos.size())

        return self.position_encoding(input_pos)


# Implements FFN equation.
class PositionalWiseFeedForward(nn.Module):
    """
    Two linear transformer and one ReLU. In addition to attention sub-layers,
    each of the layers in our encoder and decoder contains a fully connected feed-forward network,
    which is applied to each position separately and identically.
    This consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, model_dim=512, ffn_dim=2048, dropout=0.0):
        super(PositionalWiseFeedForward, self).__init__()
        self.w1 = nn.Conv1d(model_dim, ffn_dim, 1)
        self.w2 = nn.Conv1d(ffn_dim, model_dim, 1)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(model_dim)

    def forward(self, x):
        output = x.transpose(1, 2)
        output = self.w2(func.relu(self.w1(output)))
        output = self.dropout(output.transpose(1, 2))
        # add residual and norm layer
        output = self.layer_norm(x + output)

        return output


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# One layer of Encoder
class EncoderLayer(nn.Module):
    """
    Core encoder is a stack of N layers.
    """
    def __init__(self, model_dim=512, num_heads=8, ffn_dim=2018, dropout=0.0):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, attn_mask=None):
        """
        Pass the input (and mask) through each layer in turn.
        """
        context, attention = self.attention(inputs, inputs, inputs, attn_mask)
        output = self.feed_forward(context)

        return output, attention


class Encoder(nn.Module):
    """
    Encoder is made up of self-attn and feed forward.
    """
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList(
          [EncoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        # size(1) of word embedding and positional embedding is model_dim
        # inputs_clone = inputs.clone()
        inputs = inputs.clone().detach().long()
        # print(inputs.size())
        output = self.seq_embedding(inputs)
        # print(output.size(), inputs.size())
        output += self.pos_embedding(inputs_len)
        self_attention_mask = padding_mask(inputs, inputs)

        attentions = []
        for encoder in self.encoder_layers:
            output, attention = encoder(output, self_attention_mask)
            attentions.append(attention)
            pass
        pass

        return output, attentions


# One layer of Decoder
class DecoderLayer(nn.Module):
    """
    Core decoder is a stack of N layers.
    """
    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(DecoderLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, dec_inputs, enc_outputs, self_attn_mask=None, context_attn_mask=None):
        # self attention, all inputs are decoder inputs as key, value and query
        dec_output, self_attention = self.attention(dec_inputs, dec_inputs, dec_inputs, self_attn_mask)
        # query is decoder's outputs, key and value are encoder's inputs
        dec_output, context_attention = self.attention(enc_outputs, enc_outputs, dec_output, context_attn_mask)
        dec_output = self.feed_forward(dec_output)

        return dec_output, self_attention, context_attention


class Decoder(nn.Module):
    """
    Decoder is made of self-attn, src-attn, and feed forward.
    """
    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(Decoder, self).__init__()
        self.num_layers = num_layers
        self.decoder_layers = nn.ModuleList(
          [DecoderLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.seq_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len, enc_output, context_attn_mask=None):
        inputs = inputs.clone().detach().long()
        output = self.seq_embedding(inputs)
        output += self.pos_embedding(inputs_len)
        self_attention_padding_mask = padding_mask(inputs, inputs)
        seq_mask = sequence_mask(inputs)
        if torch.cuda.is_available():
            self_attn_mask = torch.gt((self_attention_padding_mask.cuda() + seq_mask.cuda()), 0)
            pass
        else:
            self_attn_mask = torch.gt((self_attention_padding_mask + seq_mask), 0)
            pass
        pass

        self_attentions = []
        context_attentions = []
        for decoder in self.decoder_layers:
            output, self_attn, context_attn = decoder(output, enc_output, self_attn_mask, context_attn_mask)
            self_attentions.append(self_attn)
            context_attentions.append(context_attn)
            pass
        pass

        return output, self_attentions, context_attentions


# A sequence to sequence model with attention mechanism.
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, tgt_vocab_size, tgt_max_len,
                 num_layers=6, model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Transformer, self).__init__()
        self.encoder = Encoder(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.decoder = Decoder(tgt_vocab_size, tgt_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, tgt_vocab_size, bias=False)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, src_seq, src_len, tgt_seq, tgt_len):
        context_attn_mask = padding_mask(tgt_seq, src_seq)
        output, enc_self_attn = self.encoder(src_seq, src_len)
        output, dec_self_attn, ctx_attn = self.decoder(tgt_seq, tgt_len, output, context_attn_mask)

        output = self.linear(output)
        output = self.softmax(output)

        return output, enc_self_attn, dec_self_attn, ctx_attn


# A simple wrapper class for learning rate scheduling.
class ScheduledOptim(nn.Module):
    def __init__(self, optimizer, d_model, n_warmup_steps=4000):
        super(ScheduledOptim, self).__init__()
        self._optimizer = optimizer
        self.n_warmup_steps = n_warmup_steps
        self.n_current_steps = 0
        self.init_lr = np.power(d_model, -0.5)

    # Step with the inner optimizer.
    def step_and_update_lr(self):
        self._update_learning_rate()
        self._optimizer.step()

    # Zero out the gradients by the inner optimizer
    def zero_grad(self):
        self._optimizer.zero_grad()

    def _get_lr_scale(self):
        return np.min([
            np.power(self.n_current_steps, -0.5),
            np.power(self.n_warmup_steps, -1.5) * self.n_current_steps])

    # Learning rate scheduling per step.
    def _update_learning_rate(self):
        self.n_current_steps += 1
        lr = self.init_lr * self._get_lr_scale()
        for param_group in self._optimizer.param_groups:
            param_group['lr'] = lr
            pass
        pass


if __name__ == '__main__':
    temp_input = torch.zeros((64, 62))
    temp_target = torch.zeros((64, 62))
    lens_input = torch.ones(64).long()
    lens_target = torch.ones(64).long()
    lens_input[:] = 62
    lens_target[:] = 62

    sample_transformer = Transformer(8000, 64, 8000, 64)
    fn_out, _, _, _ = sample_transformer(temp_input, lens_input, temp_target, lens_target)

    print(fn_out.size())  # (batch_size, tar_seq_len, target_vocab_size)
