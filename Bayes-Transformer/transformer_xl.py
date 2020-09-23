import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as func

from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout


def _rel_shift(x, zero_triu=False):
    # x: [word_len, word_len + 1, batch_size, num_heads]
    x_padded = x.reshape(x.size(1), x.size(0), *x.size()[2:])
    x = x_padded[1:].reshape(x.size(0), x.size(1) - 1, *x.size()[2:])

    if zero_triu:
        ones = torch.ones((x.size(0), x.size(1)))
        x = x * torch.tril(ones, x.size(1) - x.size(0))[:, :, None, None]

    return x


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.inv = inv_freq
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        # print(self.inv)
        # print(pos_seq.size(), self.inv.size())
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        # print(sinusoid_inp.size())
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
        # print(pos_emb.size())

        if bsz is not None:
            return pos_emb[:, None, :].expand(-1, bsz, -1)
        else:
            return pos_emb[:, None, :]


class BayesEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, padding_idx=None,
                 max_norm=None, norm_type=2., scale_grad_by_freq=False,
                 sparse=False, _weight=None):
        super(BayesEmbedding, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        if padding_idx is not None:
            if padding_idx > 0:
                assert padding_idx < self.num_embeddings, 'Padding_idx must be within num_embeddings'
            elif padding_idx < 0:
                assert padding_idx >= -self.num_embeddings, 'Padding_idx must be within num_embeddings'
                padding_idx = self.num_embeddings + padding_idx
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.scale_grad_by_freq = scale_grad_by_freq
        self.sparse = sparse
        self.weight_mean = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim), requires_grad=True)
        self.weight_lgstd = nn.Parameter(torch.Tensor(num_embeddings, embedding_dim), requires_grad=True)
        self.weight = torch.Tensor(num_embeddings, embedding_dim)
        self.reset_parameters()

    def reset_parameters(self):
        stde = 1. / math.sqrt(self.embedding_dim + 1)
        self.weight_mean.data.uniform_(-stde, stde)
        self.weight_lgstd.data.uniform_(2*np.log(stde), 1*np.log(stde))
        if self.padding_idx is not None:
            with torch.no_grad():
                self.weight[self.padding_idx].fill_(0)

    def sample_weight_diff(self):
        if self.training:
            weight_std = torch.exp(self.weight_lgstd)
            epsilon = weight_std.new_zeros(*weight_std.size()).normal_()
            weight_diff = epsilon * weight_std
            return weight_diff
        return 0

    def kl_divergence(self):
        kl = 0
        theta_mean = self.weight_mean
        theta_std = self.weight_lgstd
        kl += torch.mean(theta_mean ** 2. - theta_std * 2. + torch.exp(theta_std * 2)) / 2.
        return kl

    def forward(self, inputs):
        self.weight = self.weight_mean*1.
        weight_diff = self.sample_weight_diff()
        self.weight += weight_diff
        return func.embedding(inputs, self.weight, self.padding_idx)


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.0):
        super(PositionwiseFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Linear(d_model, d_inner), nn.ReLU(inplace=True),
            LockedDropout(dropout),
            nn.Linear(d_inner, d_model),
            LockedDropout(dropout),
        )

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, display=False):
        # positionwise feed-forward
        core_out = self.CoreNet(inp)

        # residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output, 0


class BayesPositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner):
        super(BayesPositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.weight_mean1 = nn.Parameter(torch.rand(self.d_inner, self.d_model), requires_grad=True)
        self.weight_mean2 = nn.Parameter(torch.rand(self.d_model, self.d_inner), requires_grad=True)
        self.weight_std1 = nn.Parameter(torch.rand(self.d_inner, self.d_model), requires_grad=True)
        self.weight_std2 = nn.Parameter(torch.rand(self.d_model, self.d_inner), requires_grad=True)
        self.reset_parameters()
        self.layer_norm = nn.LayerNorm(d_model)

    def reset_parameters(self):
        stdm = 1. / math.sqrt(self.d_model)
        stdi = 1. / math.sqrt(self.d_inner)
        self.weight_mean1.data.uniform_(-stdm, stdm)
        self.weight_mean2.data.uniform_(-stdi, stdi)
        self.weight_std1.data.uniform_(2*np.log(stdm), 1*np.log(stdm))
        self.weight_std2.data.uniform_(2*np.log(stdi), 1*np.log(stdi))

    def sample_weight_diff(self):
        if self.training:
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

    def forward(self, inp, display=False):
        # positionwise feed-forward
        # print("inp.size(): ", inp.size())
        weight1 = self.weight_mean1*1.
        weight2 = self.weight_mean2*1.
        weight1_diff, weight2_diff = self.sample_weight_diff()
        if display is True:
            print("weight1, weight1_diff, weight2, weight2_diff:",
                  weight1.mean().item(), weight1_diff.mean().item(),
                  weight2.mean().item(), weight2_diff.mean().item())
        weight1 += weight1_diff
        weight2 += weight2_diff
        # if display is True:
        #     print("weight1, weight1_diff, weight2, weight2_diff:",
        #           weight1.mean().item(), weight1_diff.mean().item(),
        #           weight2.mean().item(), weight2_diff.mean().item())

        # print("weight1_size:", self.weight1.size())
        # print("weight2_size:", self.weight2.size())
        layer1_out = func.linear(inp, weight1)
        # print("layer1_out_size:", layer1_out.size())
        layer2_out = func.linear(layer1_out, weight2)
        # print("layer2_out_size:", layer2_out.size())

        # residual connection + layer normalization
        output = self.layer_norm(inp + layer2_out)

        ff_kl = self.kl_divergence()
        # kl = 0

        return output, ff_kl


class MultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head, dropout):
        super(MultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.dropout = dropout

        self.qkv_net = nn.Sequential(
            nn.Linear(d_model, 3 * n_head * d_head, bias=False),
            LockedDropout(dropout)
        )

        self.drop = nn.Dropout(dropout)
        self.locked_drop = LockedDropout(dropout)

        self.o_net = nn.Linear(n_head * d_head, d_model, bias=False)

        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def forward(self, h, attn_mask=None, pad_mask=None, mems=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        head_q, head_k, head_v = torch.chunk(self.qkv_net(h), 3, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_v = head_v.view(h.size(0), h.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', [head_q, head_k])
        attn_score.mul_(self.scale)
        # print("attn_score.size: ", attn_score.size())
        # print("dec_attn_mask.size: ", attn_mask.size(), pad_mask.size())

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
                # print("dec_attn_mask.size: ", attn_mask[:, :, :, None].size())
                # print("dec_attn_mask.size: ", attn_mask)

        # if not self.training:
        #     # print("attn_mask: ", attn_mask[:, :, 0])
        #     # print("pad_mask: ", pad_mask[:, :, 0])
        #     # print("attn_mask.size: ", attn_mask.size())
        #     # print("pad_mask.size: ", pad_mask.size())
        #     # dec_attn_mask = torch.gt(attn_mask + pad_mask, 0)
        #     # print("dec_attn_mask.size: ", attn_mask.size())
        #     # print("dec_attn_mask: ", dec_attn_mask[:, :, 0])
        #     # print("attn_score.size: ", attn_score.size())
        #     # attn_score.masked_fill_(pad_mask[:, :, :, None], -float('inf'))
        #     pass
        # else:
        #     pass

        # [qlen x klen x bsz x n_head]
        attn_prob = func.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', [attn_prob, head_v])
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.locked_drop(attn_out)

        # residual connection + layer normalization
        output = self.layer_norm(h + attn_out)

        return output, 0


class BayesMultiHeadAttn(nn.Module):
    def __init__(self, n_head, d_model, d_head):
        super(BayesMultiHeadAttn, self).__init__()

        self.n_head = n_head
        self.d_model = d_model
        self.d_head = d_head

        self.query_mean = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)
        self.key_mean = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)
        self.value_mean = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)
        self.query_std = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)
        self.key_std = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)
        self.value_std = nn.Parameter(torch.rand(self.n_head * self.d_head, self.d_model), requires_grad=True)

        self.output_mean = nn.Parameter(torch.rand(self.d_model, self.n_head * self.d_head), requires_grad=True)
        self.output_std = nn.Parameter(torch.rand(self.d_model, self.n_head * self.d_head), requires_grad=True)
        self.reset_parameters()
        self.layer_norm = nn.LayerNorm(d_model)

        self.scale = 1 / (d_head ** 0.5)

    def reset_parameters(self):
        init_qkv = 1. / math.sqrt(self.d_model)
        init_out = 1. / math.sqrt(self.n_head * self.d_head)

        # Initial function. Given expected values. While mean~U(-u, u), std~U(1/w^2, 1/w)
        self.query_mean.data.uniform_(-init_qkv, init_qkv)
        self.query_std.data.uniform_(2*np.log(init_qkv), np.log(init_qkv))
        self.key_mean.data.uniform_(-init_qkv, init_qkv)
        self.key_std.data.uniform_(2*np.log(init_qkv), np.log(init_qkv))
        self.value_mean.data.uniform_(-init_qkv, init_qkv)
        self.value_std.data.uniform_(2*np.log(init_qkv), np.log(init_qkv))

        self.output_mean.data.uniform_(-init_out, init_out)
        self.output_std.data.uniform_(2*np.log(init_out), np.log(init_out))

    def sample_weight_diff(self):
        if self.training:
            # Sampling function, from sample = mean + N(0, 1) * exp (lgstd)
            # lgstd_list = [self.query_std, self.key_std, self.value_std, self.output_std]
            # sample_list = []
            # for lgstd in lgstd_list:
            #     variance = torch.exp(lgstd)
            #     epsilon = variance.new_zeros(*variance.size()).normal_()
            #     sample = epsilon * variance
            #     sample_list.append(sample)
            #     pass
            # pass
            #
            # query_sample = sample_list[0]
            # key_sample = sample_list[1]
            # value_sample = sample_list[2]
            # output_sample = sample_list[3]

            query_var = torch.exp(self.query_std)
            epsilon = query_var.new_zeros(*query_var.size()).normal_()
            query_sample = epsilon*query_var
            key_var = torch.exp(self.key_std)
            epsilon = key_var.new_zeros(*key_var.size()).normal_()
            key_sample = epsilon*key_var
            value_var = torch.exp(self.value_std)
            epsilon = value_var.new_zeros(*value_var.size()).normal_()
            value_sample = epsilon*value_var
            output_var = torch.exp(self.output_std)
            epsilon = output_var.new_zeros(*output_var.size()).normal_()
            output_sample = epsilon*output_var

            return query_sample, key_sample, value_sample, output_sample
        return 0, 0, 0, 0

    def kl_divergence(self):
        kl = 0
        theta_mean = torch.cat([self.query_mean, self.key_mean], -1)
        theta_mean = torch.cat([theta_mean, self.value_mean], -1)
        theta_mean = torch.cat([theta_mean, self.output_mean.t()], -1)
        theta_std = torch.cat([self.query_std, self.key_std], -1)
        theta_std = torch.cat([theta_std, self.value_std], -1)
        theta_std = torch.cat([theta_std, self.output_std.t()], -1)
        kl += torch.mean(theta_mean ** 2. - theta_std * 2. + torch.exp(theta_std * 2)) / 2.
        return kl

    def forward(self, dec_inp, attn_mask=None, pad_mask=None, mems=None):
        # multihead attention
        # [hlen x bsz x n_head x d_head]
        # Get the weight for query, key and value
        query_weight = self.query_mean * 1.
        key_weight = self.key_mean * 1.
        value_weight = self.value_mean * 1.
        query_diff, key_diff, value_diff, output_diff = self.sample_weight_diff()
        query_weight += query_diff
        key_weight += key_diff
        value_weight += value_diff

        head_q = func.linear(dec_inp, query_weight)
        head_k = func.linear(dec_inp, key_weight)
        head_v = func.linear(dec_inp, value_weight)
        head_q = head_q.view(dec_inp.size(0), dec_inp.size(1), self.n_head, self.d_head)
        head_k = head_k.view(dec_inp.size(0), dec_inp.size(1), self.n_head, self.d_head)
        head_v = head_v.view(dec_inp.size(0), dec_inp.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', [head_q, head_k])
        attn_score.mul_(self.scale)
        # print("attn_score.size: ", attn_score.size())
        # print("dec_attn_mask.size: ", attn_mask.size(), pad_mask.size())

        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = func.softmax(attn_score, dim=1)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head] -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', [attn_prob, head_v])
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        output_weight = self.value_mean * 1.
        output_weight += output_diff
        attn_out = func.linear(attn_vec, output_weight)

        # residual connection + layer normalization
        output = self.layer_norm(dec_inp + attn_out)
        attn_kl = self.kl_divergence()

        return output, attn_kl


class RelMultiHeadAttn(MultiHeadAttn):
    def __init__(self, n_head, d_model, d_head, dropout):
        super(RelMultiHeadAttn, self).__init__(n_head, d_model, d_head, dropout)

    def forward(self, w, r, attn_mask=None, mems=None):
        # w: dec_inp, r: pos_emb.
        # print("w.size(), r.size(): ", w.size(), r.size())
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            w_heads = self.qkv_net(torch.cat([mems, w], 0))
            r_heads = self.qkv_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            # generate query, key, value about dec_input by linear transform
            # print("w.size(): ", w.size())
            w_heads = self.qkv_net(w)
            # print("w_heads.size(): ", w_heads.size())
            # print("r.size(): ", r.size())
            r_heads = self.qkv_net(r)
            # print("r_heads.size(): ", r_heads.size())

            # split the query, key, value.
            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)

        klen = w_head_k.size(0)
        # split q, k, v using multi-head attention
        # print("w_head_q: ", w_head_q.size())
        # print("qlen, klen: ", qlen, klen), qlen = klen
        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        # print("w_head_q.reshape: ", w_head_q.size())
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_q = r_head_q.view(rlen, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        # compute attention score
        # i: num_word, j: num_pos, b: batch, n: n_heads, d: d_heads.
        # print("w_head_q + r_head_q[-1]: ", w_head_q.size(), r_head_q.size(), r_head_q[-1].size())
        rw_head_q = w_head_q + r_head_q[-1]
        # print("rw_head_q.size(): ", rw_head_q.size())
        ac_rw_qk = torch.einsum('ibnd,jbnd->ijbn', [rw_head_q, w_head_k])

        rr_head_q = w_head_q + r_head_q[-1]
        # print("rr_head_q.size(), r_head_k.size(): ", rr_head_q.size(), r_head_k.size())
        bd_rr_qk = torch.einsum('ibnd,jnd->ijbn', [rr_head_q, r_head_k])
        # print("BD.size(): ", BD.size())
        # change [word_len, word_len + 1, batch_size, num_heads] to [word_len, word_len, batch_size, num_heads]
        bd_rr_qk = _rel_shift(bd_rr_qk)
        # print("BD.size_shift(): ", BD.size())

        # [qlen x klen x bsz x n_head]
        attn_score = ac_rw_qk + bd_rr_qk
        # print(self.scale)
        attn_score.mul_(self.scale)

        # compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None, :, :, None], -float('inf'))
                # print("attn_mask 2")
            elif attn_mask.dim() == 3:
                # print("attn_mask.size(): ", attn_mask.size())
                # print("attn_mask", attn_mask[:,:,:,None].size())
                attn_score.masked_fill_(attn_mask[:, :, :, None], -float('inf'))
                # print("attn_mask 3")

        # [qlen x klen x bsz x n_head]
        # print("attn_score.size(): ", attn_score.size())
        attn_prob = func.softmax(attn_score, dim=1)
        # print("attn_prob.size(): ", attn_prob.size())
        attn_prob = self.drop(attn_prob)
        # attn_prob = self.locked_drop(attn_prob)

        # compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', [attn_prob, w_head_v])

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.locked_drop(attn_out)

        # residual connection + layer normalization
        output = self.layer_norm(w + attn_out)

        return output


class RelDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta, bayes_ffn=None, bayes_attn=None,
                 **kwargs):
        super(RelDecoderLayer, self).__init__()
        if bayes_attn is True:
            self.dec_attn = BayesMultiHeadAttn(n_head, d_model, d_head, **kwargs)
        else:
            self.dec_attn = MultiHeadAttn(n_head, d_model, d_head, dropouta, **kwargs)
        pass

        if bayes_ffn is True:
            self.pos_ff = BayesPositionwiseFF(d_model, d_inner)
        else:
            self.pos_ff = PositionwiseFF(d_model, d_inner, dropoutf)
        pass

    def forward(self, dec_inp, pos_emb, dec_attn_mask=None, pad_mask=None, mems=None, display=False):
        output, kl_attn = self.dec_attn(dec_inp, attn_mask=dec_attn_mask, pad_mask=pad_mask, mems=mems)
        output, kl_ff = self.pos_ff(output, display)
        kl = kl_ff + kl_attn

        return output, kl


class AWDTransformerXL(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropoute, dropouti, dropouta, dropoutf, dropouth, dropouto,
                 tie_weight=True, tgt_len=None, ext_len=0, mem_len=0, clamp_len=-1,
                 bayes_ffn=0, bayes_attn=0, bayes_embed=False):
        super(AWDTransformerXL, self).__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head
        self.bayes_embed = bayes_embed

        if self.bayes_embed is True:
            self.word_emb = BayesEmbedding(n_token, d_model).cuda()
        else:
            self.word_emb = nn.Embedding(n_token, d_model)
        pass

        self.emb_scale = d_model ** 0.5

        self.dropoute = dropoute
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropouto = dropouto

        self.drop_i = nn.Dropout(dropouti)
        self.locked_drop_i = LockedDropout(dropouti)
        self.locked_drop_h = LockedDropout(dropouth)
        self.locked_drop_o = LockedDropout(dropouto)

        self.n_layer = n_layer

        self.tgt_len = tgt_len
        self.ext_len = ext_len
        self.mem_len = mem_len
        self.clamp_len = clamp_len

        self.layers = nn.ModuleList()
        for loop_i in range(n_layer):
            if loop_i + 1 <= bayes_ffn and loop_i + 1 <= bayes_attn:
                self.layers.append(
                    RelDecoderLayer(n_head, d_model, d_head, d_inner, dropoutf=dropoutf,
                                    dropouta=dropouta, bayes_ffn=True, bayes_attn=True)
                )
                pass
            elif bayes_ffn < loop_i + 1 <= bayes_attn:
                self.layers.append(
                    RelDecoderLayer(n_head, d_model, d_head, d_inner, dropoutf=dropoutf,
                                    dropouta=dropouta, bayes_ffn=False, bayes_attn=True)
                )
                pass
            elif bayes_attn < loop_i + 1 <= bayes_ffn:
                self.layers.append(
                    RelDecoderLayer(n_head, d_model, d_head, d_inner, dropoutf=dropoutf,
                                    dropouta=dropouta, bayes_ffn=True, bayes_attn=False)
                )
                pass
            else:
                self.layers.append(
                    RelDecoderLayer(n_head, d_model, d_head, d_inner, dropoutf=dropoutf,
                                    dropouta=dropouta, bayes_ffn=False, bayes_attn=False)
                )
                pass
            pass

        if self.bayes_embed:
            self.out_mean = nn.Parameter(torch.Tensor(n_token, d_model), requires_grad=True)
            self.out_mean = self.word_emb.weight_mean
            pass
        else:
            self.out_layer = nn.Linear(d_model, n_token)
            if tie_weight:
                self.out_layer.weight = self.word_emb.weight
                pass
            pass
        pass
        # self.out_layer = nn.Linear(d_model, n_token)

        self._create_params()

    def reset_length(self, tgt_len, ext_len, mem_len):
        self.tgt_len = tgt_len
        self.mem_len = mem_len
        self.ext_len = ext_len

    def _create_params(self):
        self.pos_emb = PositionalEmbedding(self.d_model)

    def init_mems(self):
        if self.mem_len > 0:
            mems = []

            for i in range(self.n_layer):
                empty = torch.empty(0, dtype=self.word_emb.weight.dtype,
                                    device=self.word_emb.weight.device)
                mems.append(empty)

            return mems
        else:
            return None

    def _update_mems(self, hids, mems, qlen, mlen):
        # does not deal with None
        if mems is None:
            return None

        # mems is not None
        assert len(hids) == len(mems), 'len(hids) != len(mems)'

        # There are `mlen + qlen` steps that can be cached into mems
        # For the next step, the last `ext_len` of the `qlen` tokens
        # will be used as the extended context. Hence, we only cache
        # the tokens from `mlen + qlen - self.ext_len - self.mem_len`
        # to `mlen + qlen - self.ext_len`.
        with torch.no_grad():
            new_mems = []
            end_idx = mlen + max(0, qlen - 0 - self.ext_len)
            beg_idx = max(0, end_idx - self.mem_len)
            for i in range(len(hids)):
                cat = torch.cat([mems[i], hids[i]], dim=0)
                new_mems.append(cat[beg_idx:end_idx].detach())

        return new_mems

    def forward(self, dec_inp, target, *mems, return_h=False, sentence_level=None, sent_lens=None, display=False):
        # print("dec_inp, target: ", dec_inp.size(), target.size())
        dec_inp = dec_inp[-target.size(0):]
        # print("dec_inp", dec_inp.size())

        if not mems:
            mems = self.init_mems()

        qlen, bsz = dec_inp.size()

        if self.bayes_embed is True:
            word_emb = self.word_emb(dec_inp)
        else:
            word_emb = embedded_dropout(self.word_emb, dec_inp, dropout=self.dropoute if self.training else 0)
        pass

        word_emb.mul_(self.emb_scale)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1 + mlen).bool()[:, :, None]
        # print("dec_attn_mask.size: ", dec_attn_mask[:, :, 0].size())
        # print("dec_attn_mask: ", dec_attn_mask[:, :, 0])

        if not self.training or sentence_level is True:
            # len_q = dec_inp.size(0)
            # pad_mask = dec_inp.t().eq(0)
            # # print("pad_mask: ", pad_mask.size(), pad_mask)
            # pad_mask = pad_mask.unsqueeze(1).expand(-1, len_q, -1)
            # pad_mask = pad_mask.transpose(0, 2).transpose(0, 1)
            # # print("pad_mask:", pad_mask.size())
            # # print(pad_mask[:, :, 1])
            pad_mask = None
            pass
        else:
            pad_mask = None
            pass
        pass

        hids = []

        # relative pos embedding
        # print("klen: ", klen)
        pos_seq = torch.arange(0, klen, 1.0, device=word_emb.device)
        # print("pos_seq: ", pos_seq)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)
        # print("pos_emb: ", pos_emb.size())

        # initial inputs
        core_out = self.locked_drop_i(word_emb)
        pos_emb = self.locked_drop_i(pos_emb)

        # compute hids
        kl_loss = 0
        for i, layer in enumerate(self.layers):
            # start_time = time.time()
            # save the input to each layer for memory
            hids.append(core_out)

            # current memory
            mems_i = mems[i] if mems is not None else None

            # print("core_out: ", core_out.size())
            core_out, kl = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask, pad_mask=pad_mask, display=display)
            kl_loss += kl
            # apply dropouth, if it is not the last layer
            if i < len(self.layers) - 1:
                core_out = self.locked_drop_h(core_out)

            # end_time = time.time()
            # print("decoder time once: ", end_time - start_time)

        # update memory
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        # compute loss
        hidden = core_out
        pred_hid = self.locked_drop_o(hidden)
        if self.bayes_embed is True:
            logit = func.linear(pred_hid, self.out_mean)
            pass
        else:
            logit = self.out_layer(pred_hid)
            pass
        pass
        # logit = self.out_layer(pred_hid)
        # print("logit.size, target.size:", logit.size(), target.size())
        # if not self.training and sentence_level is True:
        #     logit = logit[:-1, :, :]
        #     target = target[:-1, :]

        if hasattr(self, 'temperature'):
            logit = logit / self.temperature
        loss = -func.log_softmax(logit, dim=-1) \
            .gather(2, target.unsqueeze(2)).squeeze(2)

        ret = [loss]

        # return values (as a list)
        if new_mems is not None:
            ret = ret + new_mems

        # only return the last-layer hidden to reduce multi-gpu communication
        if return_h:
            ret = ret + [hidden]
        # print(kl_loss)

        ret = ret + [kl_loss]

        return ret
