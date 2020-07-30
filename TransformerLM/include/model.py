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
        :param query: shape [B, L_q, D_q, N_heads]
        :param key: shape [B, L_k, D_k, N_heads]
        :param value: shape [B, L_v, D_v, N_heads]
        :param scale: make Softmax in low gradient
        :param attn_mask: masking tensor，shape [B, L_q, L_k]
        :return: context tensor as z and attetention tensor
        """
        # attention = torch.bmm(query, key.transpose(1, 2))
        attention = torch.einsum('bind,bjnd->bijn', (query, key))
        # print(attention)
        # print(attention.size(), attn_mask.size())
        if scale:
            attention = attention * scale
            pass
        pass

        if attn_mask is not None:
            attention = attention.masked_fill_(attn_mask[:, :, :, None], -1e9)
            pass
        pass

        # print(attention)
        attention = self.softmax(attention)
        attention = self.dropout(attention)

        # context = torch.bmm(attention, value)
        # print("attention.size(), value.size(): ", attention.size(), value.size())
        context = torch.einsum('bijn,bjnd->bind', (attention, value))

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
        self.linear_k = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_v = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)
        self.linear_q = nn.Linear(model_dim, self.dim_per_head * num_heads, bias=False)

        self.dot_product_attention = ScaledDotProductAttention(dropout)
        self.linear_final = nn.Linear(model_dim, model_dim, bias=False)
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
        key = key.view(batch_size, -1, num_heads, dim_per_head)
        value = value.view(batch_size, -1, num_heads, dim_per_head)
        query = query.view(batch_size, -1, num_heads, dim_per_head)

        # print(attn_mask)
        #         if attn_mask is not None:
        #             attn_mask = attn_mask.repeat(num_heads, 1, 1)
        #             pass
        #         pass

        # scaled dot-product attention
        scale = key.size(-1) ** -0.5
        context, attention = self.dot_product_attention(query, key, value, scale, attn_mask)
        # concat heads
        context = context.contiguous().view(batch_size, context.size(1), dim_per_head * num_heads)
        output = self.linear_final(context)
        output = self.dropout(output)

        # add residual and norm layer
        output = self.layer_norm(residual + output)

        return output, attention


# Make paddings in sequences invisible.
def padding_mask(seq_k, seq_q):
    """
    :param seq_k: shape [B, L]
    :param seq_q: shape [B, L]
    :return: pad_mask shape [B, L, L]
    """
    len_q = seq_q.size(1)
    pad_mask = seq_k.eq(0)
    # print("pad_mask: ", pad_mask.size(), pad_mask)
    pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)

    return pad_mask


# To make latter word invisible for present decoder.
def sequence_mask(seq):
    """
    :param seq: shape [B, L]
    :return: mask shape [B, L, L]
    """
    batch_size, seq_len = seq.size()
    # mask = torch.zeros((seq_len, seq_len))
    mask = torch.triu(torch.ones((seq_len, seq_len), dtype=torch.uint8), diagonal=1)
    mask = mask.unsqueeze(0).expand(batch_size, -1, -1)

    return mask


# # Compute the positional encodings once in trigonometric space.
# class PositionalEncoding(nn.Module):
#     """
#     Where pos is the position and i is the dimension.
#     That is, each dimension of the positional encoding corresponds to a sinusoid.
#     The wavelengths form a geometric progression from 2π to 10000 · 2π.
#     We chose this function because we hypothesized
#     it would allow the model to easily learn to attend by relative positions,
#     since for any ﬁxed offset k, PE pos+k can be represented as a linear function of PE pos .
#     """
#     def __init__(self, d_model, max_len=150):
#         super(PositionalEncoding, self).__init__()
#         self.encoding = torch.zeros(max_len, d_model)
#         self.encoding.requires_grad = False

#         pos = torch.arange(0, max_len)
#         pos = pos.float().unsqueeze(dim=1)

#         _2i = torch.arange(0, d_model, 2).float()

#         self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_model)))
#         self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_model)))

#     def forward(self, x):
#         batch_size, seq_len = x.size()
#         # print("seq_len: ", seq_len)
#         return self.encoding[:seq_len, :].cuda()


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

    def __init__(self, d_model, max_seq_len=150):
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
        output = self.w2(func.relu(self.w1(output), inplace=True))
        output = self.dropout(output.transpose(1, 2))
        # add residual and norm layer
        output = self.layer_norm(x + output)

        return output


# One layer of Decoder.
class SubLayer(nn.Module):
    """
    Decoder is made of self-attn, and feed forward.
    """

    def __init__(self, model_dim, num_heads=8, ffn_dim=2048, dropout=0.0):
        super(SubLayer, self).__init__()
        self.attention = MultiHeadAttention(model_dim, num_heads, dropout)
        self.feed_forward = PositionalWiseFeedForward(model_dim, ffn_dim, dropout)

    def forward(self, inputs, self_attn_mask=None):
        """
        Pass the input (and mask) through each layer in turn.
        self attention, all inputs as key, value and query
        """
        attn_outputs, self_attention = self.attention(inputs, inputs, inputs, self_attn_mask)
        outputs = self.feed_forward(attn_outputs)

        return outputs, self_attention


class MainLayer(nn.Module):
    """
    Core decoder is a stack of N layers.
    """

    def __init__(self, vocab_size, max_seq_len, num_layers=6, model_dim=512,
                 num_heads=8, ffn_dim=2048, dropout=0.0):
        super(MainLayer, self).__init__()
        self.num_layers = num_layers
        self.d_model = model_dim
        self.sublayers = nn.ModuleList(
            [SubLayer(model_dim, num_heads, ffn_dim, dropout) for _ in range(num_layers)])
        self.word_embedding = nn.Embedding(vocab_size + 1, model_dim, padding_idx=0)
        self.pos_embedding = PositionalEncoding(model_dim, max_seq_len)

    def forward(self, inputs, inputs_len):
        inputs = inputs.clone().detach().long()
        embeddings = self.word_embedding(inputs)
        embeddings = embeddings * np.sqrt(self.d_model)
        embeddings += self.pos_embedding(inputs_len)
        seq_mask = sequence_mask(inputs)
        pad_mask = padding_mask(inputs, inputs)

        # print(seq_mask, self_attention_padding_mask)
        if torch.cuda.is_available():
            attn_mask = torch.gt(seq_mask.cuda() + pad_mask.cuda(), 0)
            pass
        else:
            attn_mask = torch.gt(seq_mask + pad_mask, 0)
            pass
        pass

        # print(self_attn_mask)
        self_attentions = []
        output = embeddings
        for sublayer in self.sublayers:
            # start_time = time.time()
            output, attention = sublayer(output, attn_mask)
            self_attentions.append(attention)
            # end_time = time.time()
            # print("decoder time once: ", end_time - start_time)
            pass
        pass

        return output, self_attentions


# A sequence to sequence model with attention mechanism.
class Modeling(nn.Module):
    def __init__(self, src_vocab_size, src_max_len, num_layers=6,
                 model_dim=512, num_heads=8, ffn_dim=2048, dropout=0.2):
        super(Modeling, self).__init__()
        self.layers = MainLayer(src_vocab_size, src_max_len, num_layers, model_dim, num_heads, ffn_dim, dropout)
        self.linear = nn.Linear(model_dim, src_vocab_size, bias=False)

    def forward(self, input_seq, input_len):
        output, self_attention = self.layers(input_seq, input_len)
        output = self.linear(output)

        return output, self_attention


if __name__ == '__main__':
    temp_input = torch.zeros((8, 62))
    lens_input = torch.ones(8).long()
    lens_input[:] = 62

    test_model = Modeling(8000, 62)
    fn_out, _ = test_model(temp_input, lens_input)

    print(fn_out.size())  # (batch_size, tar_seq_len, target_vocab_size)
