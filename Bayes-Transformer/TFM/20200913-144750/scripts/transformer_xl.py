import functools
import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_softmax import AdaptiveLogSoftmax
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout


def _rel_shift(x, zero_triu=False):
    x_padded = x.reshape(x.size(1), x.size(0), *x.size()[2:])
    x = x_padded[1:].reshape(x.size(0), x.size(1) - 1, *x.size()[2:])

    if zero_triu:
        ones = torch.ones((x.size(0), x.size(1)))
        x = x * torch.tril(ones, x.size(1) - x.size(0))[:,:,None,None]

    return x


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()

        self.demb = demb

        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.ger(pos_seq, self.inv_freq)
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)

        if bsz is not None:
            return pos_emb[:,None,:].expand(-1, bsz, -1)
        else:
            return pos_emb[:,None,:]


class PositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
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

    def forward(self, inp):
        ##### positionwise feed-forward
        core_out = self.CoreNet(inp)

        ##### residual connection + layer normalization
        output = self.layer_norm(inp + core_out)

        return output


class BayesPositionwiseFF(nn.Module):
    def __init__(self, d_model, d_inner, dropout):
        super(BayesPositionwiseFF, self).__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout
        self.weight_mean1 = nn.Parameter(torch.rand(self.d_inner, self.d_model))
        self.weight_mean2 = nn.Parameter(torch.rand(self.d_model, self.d_inner))
        self.weight_std1 = nn.Parameter(torch.rand(self.d_inner, self.d_model))
        self.weight_std2 = nn.Parameter(torch.rand(self.d_model, self.d_inner))
        self.reset_parameters()
        self.layer_norm = nn.LayerNorm(d_model)

    def reset_parameters(self):
        stdm = 1. / math.sqrt(self.d_model+1)
        stdi = 1. / math.sqrt(self.d_inner+1)
        print("stdm, stdi", stdm, stdi)
        self.weight_mean1.data.uniform_(-stdm, stdm)
        self.weight_mean2.data.uniform_(-stdi, stdi)
        print("log.stdm, stdi", 2*np.log(stdm), np.log(stdm))
        self.weight_std1.data.uniform_(2*np.log(stdm), np.log(stdm))
        self.weight_std2.data.uniform_(2*np.log(stdi), np.log(stdi))

    def sample_weight_diff(self):
        if self.training:
            weight_std_1 = torch.log(self.weight_std1)
            epsilon = weight_std_1.new_zeros(*weight_std_1.size()).normal_()
            # print("epsilon", epsilon)
            weight_diff_1 = epsilon*weight_std_1
            weight_std_2 = torch.log(self.weight_std2)
            epsilon = weight_std_2.new_zeros(*weight_std_2.size()).normal_()
            weight_diff_2 = epsilon*weight_std_2
            return weight_diff_1, weight_diff_2
        return 0, 0

    def kl_divergence(self):
        kl = 0
        theta_mean = torch.cat([self.weight_mean1, self.weight_mean2.t()], -1)
        theta_std = torch.cat([self.weight_std1, self.weight_std2.t()], -1)
        kl += torch.mean(theta_mean ** 2. - theta_std * 2. + torch.exp(theta_std * 2)) / 2.
        # kl += torch.mean(theta_mean ** 2. - theta_mean * 2. + theta_std ** 2 + 2 * torch.log(theta_std)) / 2.
        return kl

    def forward(self, inp):
        ##### positionwise feed-forward
        # print("inp.size(): ", inp.size())
        self.weight1 = self.weight_mean1*1.
        self.weight2 = self.weight_mean2*1.
        weight1_diff, weight2_diff = self.sample_weight_diff()
        self.weight1 += weight1_diff
        self.weight2 += weight2_diff
        # print("weight1_size:", self.weight1.size())
        # print("weight2_size:", self.weight2.size())
        layer1_out = F.linear(inp, self.weight1)
        # print("layer1_out_size:", layer1_out.size())
        layer2_out = F.linear(F.relu(layer1_out) ,self.weight2)
        # print("layer2_out_size:", layer2_out.size())

        ##### residual connection + layer normalization
        output = self.layer_norm(inp + layer2_out)

        kl = self.kl_divergence()

        return output, kl


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

    def forward(self, h, attn_mask=None, mems=None):
        ##### multihead attention
        # [hlen x bsz x n_head x d_head]
        head_q, head_k, head_v = torch.chunk(self.qkv_net(h), 3, -1)
        head_q = head_q.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_k = head_k.view(h.size(0), h.size(1), self.n_head, self.d_head)
        head_v = head_v.view(h.size(0), h.size(1), self.n_head, self.d_head)

        # [qlen x klen x bsz x n_head]
        attn_score = torch.einsum('ibnd,jbnd->ijbn', (head_q, head_k))
        attn_score.mul_(self.scale)
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)

        # [qlen x klen x bsz x n_head] + [klen x bsz x n_head x d_head]
        # -> [qlen x bsz x n_head x d_head]
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, head_v))
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.locked_drop(attn_out)

        ##### residual connection + layer normalization
        output = self.layer_norm(h + attn_out)

        return output


class RelMultiHeadAttn(MultiHeadAttn):
    def __init__(self, n_head, d_model, d_head, dropout):
        super(RelMultiHeadAttn, self).__init__(n_head, d_model, d_head, dropout)

    def forward(self, w, r, attn_mask=None, mems=None):
        qlen, rlen, bsz = w.size(0), r.size(0), w.size(1)

        if mems is not None:
            w_heads = self.qkv_net(torch.cat([mems, w], 0))
            r_heads = self.qkv_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)

            w_head_q = w_head_q[-qlen:]
        else:
            w_heads = self.qkv_net(w)
            r_heads = self.qkv_net(r)

            w_head_q, w_head_k, w_head_v = torch.chunk(w_heads, 3, dim=-1)
            r_head_q, r_head_k, r_head_v = torch.chunk(r_heads, 3, dim=-1)

        klen = w_head_k.size(0)

        w_head_q = w_head_q.view(qlen, bsz, self.n_head, self.d_head)
        w_head_k = w_head_k.view(klen, bsz, self.n_head, self.d_head)
        w_head_v = w_head_v.view(klen, bsz, self.n_head, self.d_head)

        r_head_q = r_head_q.view(rlen, self.n_head, self.d_head)
        r_head_k = r_head_k.view(rlen, self.n_head, self.d_head)

        #### compute attention score
        rw_head_q = w_head_q + r_head_q[-1]
        AC = torch.einsum('ibnd,jbnd->ijbn', (rw_head_q, w_head_k))

        rr_head_q = w_head_q + r_head_q[-1]
        BD = torch.einsum('ibnd,jnd->ijbn', (rr_head_q, r_head_k))
        BD = _rel_shift(BD)

        # [qlen x klen x bsz x n_head]
        attn_score = AC + BD
        attn_score.mul_(self.scale)

        #### compute attention probability
        if attn_mask is not None and attn_mask.any().item():
            if attn_mask.dim() == 2:
                attn_score.masked_fill_(attn_mask[None,:,:,None], -float('inf'))
            elif attn_mask.dim() == 3:
                attn_score.masked_fill_(attn_mask[:,:,:,None], -float('inf'))

        # [qlen x klen x bsz x n_head]
        attn_prob = F.softmax(attn_score, dim=1)
        attn_prob = self.drop(attn_prob)
        # attn_prob = self.locked_drop(attn_prob)

        #### compute attention vector
        attn_vec = torch.einsum('ijbn,jbnd->ibnd', (attn_prob, w_head_v))

        # [qlen x bsz x n_head x d_head]
        attn_vec = attn_vec.contiguous().view(
            attn_vec.size(0), attn_vec.size(1), self.n_head * self.d_head)

        ##### linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.locked_drop(attn_out)

        ##### residual connection + layer normalization
        output = self.layer_norm(w + attn_out)

        return output


class RelDecoderLayer(nn.Module):
    def __init__(self, n_head, d_model, d_head, d_inner, dropoutf, dropouta,
                 **kwargs):
        super(RelDecoderLayer, self).__init__()

        self.dec_attn = RelMultiHeadAttn(n_head, d_model, d_head, dropouta,
                                         **kwargs)
        self.pos_ff = BayesPositionwiseFF(d_model, d_inner, dropoutf)

    def forward(self, dec_inp, pos_emb, dec_attn_mask=None, mems=None):

        output = self.dec_attn(dec_inp, pos_emb, attn_mask=dec_attn_mask,
                               mems=mems)
        output, kl = self.pos_ff(output)

        return output, kl


class AWDTransformerXL(nn.Module):
    def __init__(self, n_token, n_layer, n_head, d_model, d_head, d_inner,
                 dropoute, dropouti, dropouta, dropoutf, dropouth, dropouto,
                 tie_weight=True, tgt_len=None, ext_len=0, mem_len=0,
                 clamp_len=-1):
        super(AWDTransformerXL, self).__init__()
        self.n_token = n_token
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_head

        self.word_emb = nn.Embedding(n_token, d_model)
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
        for i in range(n_layer):
            self.layers.append(
                RelDecoderLayer(
                    n_head, d_model, d_head, d_inner,
                    dropoutf=dropoutf, dropouta=dropouta)
            )

        self.out_layer = nn.Linear(d_model, n_token)
        if tie_weight:
            self.out_layer.weight = self.word_emb.weight

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
        if mems is None: return None

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

    def forward(self, dec_inp, target, *mems, return_h=False):
        # print("dec_inp.size(), target.size(): ", dec_inp.size(), target.size())
        if not mems: mems = self.init_mems()

        qlen, bsz = dec_inp.size()

        word_emb = embedded_dropout(
            self.word_emb, dec_inp,
            dropout=self.dropoute if self.training else 0)
        word_emb.mul_(self.emb_scale)

        mlen = mems[0].size(0) if mems is not None else 0
        klen = mlen + qlen
        dec_attn_mask = torch.triu(
            word_emb.new_ones(qlen, klen), diagonal=1+mlen).bool()[:,:,None]

        hids = []

        # relative pos embedding
        pos_seq = torch.arange(klen, -1, -1.0, device=word_emb.device)
        if self.clamp_len > 0:
            pos_seq.clamp_(max=self.clamp_len)
        pos_emb = self.pos_emb(pos_seq)

        # initial inputs
        core_out = self.locked_drop_i(word_emb)
        pos_emb = self.locked_drop_i(pos_emb)

        # compute hids
        kl_loss = 0
        for i, layer in enumerate(self.layers):
            # save the input to each layer for memory
            hids.append(core_out)

            # current memory
            mems_i = mems[i] if mems is not None else None

            core_out, kl = layer(core_out, pos_emb, dec_attn_mask=dec_attn_mask,
                             mems=mems_i)
            kl_loss += kl

            # apply dropouth, if it is not the last layer
            if i < len(self.layers) - 1:
                core_out = self.locked_drop_h(core_out)

        # update memory
        new_mems = self._update_mems(hids, mems, mlen, qlen)

        # print("core_out.size(): ", core_out.size(), -target.size(0))
        # compute loss
        hidden = core_out[-target.size(0):]
        pred_hid = self.locked_drop_o(hidden)
        # print("pred_hid.size(): ", pred_hid.size())

        logit = self.out_layer(pred_hid)
        # print("logit.size(): ", logit.size())
        if hasattr(self, 'temperature'):
            logit = logit / self.temperature
        loss = -F.log_softmax(logit, dim=-1) \
                 .gather(2, target.unsqueeze(2)).squeeze(2)

        ret = [loss]

        # return values (as a list)
        if new_mems is not None:
            ret = ret + new_mems

        # only return the last-layer hidden to reduce multi-gpu communication
        if return_h:
            ret = ret + [hidden]
        # print(kl_loss)
        kl_loss = kl_loss / dec_inp.size(0) * dec_inp.size(1)
        print("dec_inp.size:", dec_inp.size())
        ret = ret + [kl_loss]
        return ret

