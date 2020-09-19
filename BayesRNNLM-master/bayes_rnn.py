import math
import numpy as np
import torch
import torch.nn as nn

_VF = torch._C._VariableFunctions
_rnn_impls = {
    'LSTM': _VF.lstm,
    'GRU': _VF.gru,
    'RNN_TANH': _VF.rnn_tanh,
    'RNN_RELU': _VF.rnn_relu,
}


class FastBayesLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, position):
        super(FastBayesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # 1 for input gate, 2 for forget gate, 3 for cell gate, 4 for output gate
        self.position = position
        self.theta_hh_mean = nn.Parameter(torch.rand(hidden_size*4, hidden_size+1))
        self.theta_ih_mean = nn.Parameter(torch.rand(hidden_size*4, input_size+1))
        if 1 <= self.position <= 4:
            self.theta_hh_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size+1))
            self.theta_ih_lgstd = nn.Parameter(torch.rand(hidden_size, input_size+1))
        if self.position==0:
            self.theta_hh_lgstd = nn.Parameter(torch.rand(4*hidden_size, hidden_size+1))
            self.theta_ih_lgstd = nn.Parameter(torch.rand(4*hidden_size, input_size+1))

        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size+1)
        self.theta_hh_mean.data.uniform_(-stdv, stdv)
        self.theta_ih_mean.data.uniform_(-stdv, stdv)
        if 0 <= self.position <= 4:
            self.theta_hh_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
            self.theta_ih_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))

    def forward(self, x, hx):
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            x, batch_sizes = x
            result = _rnn_impls['LSTM'](x, batch_sizes, hx, self._flat_weights(), True, 1, 0., self.training, False)
        else:
            result = _rnn_impls['LSTM'](x, hx, self._flat_weights(), True, 1, 0., self.training, False, False)
        output, hidden = result[0], result[1:]

        if is_packed:
            output = torch.nn.utils.rnn.PackedSequence(output, batch_sizes)
        return output, hidden
    
    def kl_divergence(self, prior=None):
        kl = 0
        if 1 <= self.position <= 4:
            theta_mean = torch.cat([self.theta_hh_mean[(self.position-1)*self.hidden_size:self.position*self.hidden_size],
                                    self.theta_ih_mean[(self.position-1)*self.hidden_size:self.position*self.hidden_size]], -1)
            theta_lgstd = torch.cat([self.theta_hh_lgstd, self.theta_ih_lgstd], -1)
        if self.position==0:
            theta_mean = torch.cat([self.theta_hh_mean,self.theta_ih_mean], -1)
            theta_lgstd = torch.cat([self.theta_hh_lgstd, self.theta_ih_lgstd], -1)


        if prior==None and 0 <= self.position <= 4:
            kl += torch.mean(theta_mean**2.-theta_lgstd*2.+torch.exp(theta_lgstd*2))/2. # Max uses mean in orign
        else:
            if 1 <= self.position <= 4:
                prior = torch.cat([prior['rnns.0.theta_hh_mean'][(self.position-1)*self.hidden_size:self.position*self.hidden_size],
                                   prior['rnns.0.theta_ih_mean'][(self.position-1)*self.hidden_size:self.position*self.hidden_size]], -1)
            if self.position == 0:
                prior = torch.cat([prior['rnns.0.theta_hh_mean'],prior['rnns.0.theta_ih_mean']], -1)
            kl += torch.sum((theta_mean-prior)**2.-theta_lgstd*2.+torch.exp(theta_lgstd*2))/2.
        return kl

    def sample_weight_diff(self):
        if self.training:
            theta_hh_std = torch.exp(self.theta_hh_lgstd)
            epsilon = theta_hh_std.new_zeros(*theta_hh_std.size()).normal_()
            theta_hh_diff = epsilon*theta_hh_std
            theta_ih_std = torch.exp(self.theta_ih_lgstd)
            epsilon = theta_ih_std.new_zeros(*theta_ih_std.size()).normal_()
            theta_ih_diff = epsilon*theta_ih_std
            return theta_hh_diff, theta_ih_diff
        return 0, 0


    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def _flat_weights(self):
        self.theta_hh = self.theta_hh_mean*1.
        self.theta_ih = self.theta_ih_mean*1.
        if 1 <= self.position <= 4:
            theta_hh_diff, theta_ih_diff = self.sample_weight_diff()
            self.theta_hh[(self.position-1)*self.hidden_size:self.position*self.hidden_size] += theta_hh_diff
            self.theta_ih[(self.position-1)*self.hidden_size:self.position*self.hidden_size] += theta_ih_diff
        theta_hh = self.theta_hh
        return [self.theta_ih[:, :-1].contiguous(), theta_hh[:, :-1].contiguous(),
                self.theta_ih[:, -1].contiguous(), theta_hh[:, -1].contiguous()]

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.theta_hh_mean = nn.Parameter(torch.rand(hidden_size*4, hidden_size+1))
        self.theta_ih_mean = nn.Parameter(torch.rand(hidden_size*4, input_size+1))
        self._all_weights = [k for k, v in self.__dict__.items() if '_ih' in k or '_hh' in k]
        self.reset_parameters()
        
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size+1)
        self.theta_hh_mean.data.uniform_(-stdv, stdv)
        self.theta_ih_mean.data.uniform_(-stdv, stdv)

    def forward(self, x, hx):
        is_packed = isinstance(x, torch.nn.utils.rnn.PackedSequence)
        if is_packed:
            x, batch_sizes = x
            result = _rnn_impls['LSTM'](x, batch_sizes, hx, self._flat_weights(), True, 1, 0., self.training, False)
        else:
            result = _rnn_impls['LSTM'](x, hx, self._flat_weights(), True, 1, 0., self.training, False, False)
        output, hidden = result[0], result[1:]

        if is_packed:
            output = torch.nn.utils.rnn.PackedSequence(output, batch_sizes)
        return output, hidden

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        return s.format(**self.__dict__)

    def _flat_weights(self):
        self.theta_hh = self.theta_hh_mean*1.
        self.theta_ih = self.theta_ih_mean*1.
        return [self.theta_ih[:, :-1].contiguous(), self.theta_hh[:, :-1].contiguous(),
                self.theta_ih[:, -1].contiguous(), self.theta_hh[:, -1].contiguous()]


