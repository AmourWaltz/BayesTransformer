import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):

    def __init__(self, input_size, hidden_size, wdrop=0.):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wdrop = wdrop
        self.ingate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.ingate_act.register_parameter('weight_undropped', nn.Parameter(self.ingate_act[0].weight.data))
        self.forgetgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.forgetgate_act.register_parameter('weight_undropped', nn.Parameter(self.forgetgate_act[0].weight.data))
        self.cellgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Tanh()])
        self.cellgate_act.register_parameter('weight_undropped', nn.Parameter(self.cellgate_act[0].weight.data))
        self.outgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.outgate_act.register_parameter('weight_undropped', nn.Parameter(self.outgate_act[0].weight.data))
        self.cell_act = nn.Tanh()
        
    def drop_weight(self):
        ingate_weight = getattr(self.ingate_act, 'weight_undropped')
        setattr(self.ingate_act[0], 'weight',
                F.dropout(ingate_weight, p=self.wdrop, training=self.training))
        forgetgate_weight = getattr(self.forgetgate_act, 'weight_undropped')
        setattr(self.forgetgate_act[0], 'weight',
                F.dropout(forgetgate_weight, p=self.wdrop, training=self.training))
        cellgate_weight = getattr(self.cellgate_act, 'weight_undropped')
        setattr(self.cellgate_act[0], 'weight',
                F.dropout(cellgate_weight, p=self.wdrop, training=self.training))
        outgate_weight = getattr(self.outgate_act, 'weight_undropped')
        setattr(self.outgate_act[0], 'weight',
                F.dropout(outgate_weight, p=self.wdrop, training=self.training))
    
    def forward(self, emb, hidden):
        hx, _ = hidden
        outputs = []
        if(self.wdrop > 0.):
            self.drop_weight()
        for i, x in enumerate(emb):
            x_hx = torch.cat([x, hx[0]], -1)
            ingate = self.ingate_act(x_hx)
            forgetgate = self.forgetgate_act(x_hx)
            cellgate = self.cellgate_act(x_hx)
            cy = (forgetgate * cx) + (ingate * cellgate)
            outgate = self.outgate_act(x_hx)
            hy = outgate * self.cell_act(cy)
            outputs.append(hy)
            hx, cx = hy, cy
        outputs = torch.stack(outputs, 0)
        return outputs, (hx, cx)

class BayesLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1):
        super(BayesLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.uncertain_position = uncertain_position
        self.ingate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.forgetgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.cellgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Tanh()])
        self.outgate_act = nn.Sequential(*[nn.Linear(input_size+hidden_size, hidden_size), nn.Sigmoid()])
        self.cell_act = nn.Tanh()
        if(1 <= self.uncertain_position <= 4):
            self.theta_lgstd = nn.Parameter(torch.rand(input_size+hidden_size, hidden_size))
            self.init_parameters()
        
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.theta_lgstd.size(1))
        self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        if(1 <= self.uncertain_position <= 4):
            if(self.uncertain_position == 1):
                theta_mean = self.ingate_act[0].weight.t()
            elif(self.uncertain_position == 2):
                theta_mean = self.forgetgate_act[0].weight.t()
            elif(self.uncertain_position == 3):
                theta_mean = self.cellgate_act[0].weight.t()
            elif(self.uncertain_position == 4):
                theta_mean = self.outgate_act[0].weight.t()
            return torch.mean(theta_mean**2.-self.theta_lgstd*2.
                    +torch.exp(self.theta_lgstd*2))/2.
        return 0
    
    def bayes_linear(self, act, x):
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = act[0].weight.t()+epsilon*theta_std
            return act[1](F.linear(x, theta.t(), act[0].bias))
        return act(x)
        
    def forward(self, emb, hidden):
        hx, cx = hidden
        outputs = []
        for i, x in enumerate(emb):
            x_hx = torch.cat([x, hx[0]], -1)
            if(self.uncertain_position == 1 or self.uncertain_position == 2):
                ingate = self.bayes_linear(self.ingate_act, x_hx)
                forgetgate = self.bayes_linear(self.forgetgate_act, x_hx)
            else:
                ingate = self.ingate_act(x_hx)
                forgetgate = self.forgetgate_act(x_hx)
            if(self.uncertain_position == 3):
                cellgate = self.bayes_linear(self.cellgate_act, x_hx)
            else:
                cellgate = self.cellgate_act(x_hx)
            cy = (forgetgate * cx) + (ingate * cellgate)
            if(self.uncertain_position == 4):
                outgate = self.bayes_linear(self.outgate_act, x_hx)
            else:
                outgate = self.outgate_act(x_hx)
            hy = outgate * self.cell_act(cy)
            outputs.append(hy)
            hx, cx = hy, cy
        outputs = torch.stack(outputs, 0)
        outputs = torch.squeeze(outputs, 1)
        return outputs, (hx, cx)

class GPLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1):
        super(GPLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_set = {'sigmoid', 'tanh', 'relu', 'gsk'}
        self.uncertain_position = uncertain_position
        self.input_map = nn.Linear(input_size, 4*hidden_size)
        self.hidden_map = nn.Linear(hidden_size, 4*hidden_size)
        self.ingate_act = nn.Sigmoid()
        self.forgetgate_act = nn.Sigmoid()
        self.cellgate_act = nn.Tanh()
        self.outgate_act = nn.Sigmoid()
        self.cell_act = nn.Tanh()
        if(uncertain_position == 0):
            self.theta_mean = nn.Parameter(torch.rand(input_size, input_size))
            self.theta_lgstd = nn.Parameter(torch.rand(input_size, input_size))
        elif(uncertain_position == 1):
            self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        elif(uncertain_position >= 2):
            self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.basis_map = nn.Linear(hidden_size, hidden_size)
        self.init_parameters()
        
    def init_parameters(self):
        if(0 <= self.uncertain_position <= 6):
            stdv = 1. / math.sqrt(self.theta_mean.size(1))
            self.theta_mean.data.uniform_(-stdv, stdv)
            self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        if(0 <= self.uncertain_position <= 6):
            return torch.mean(self.theta_mean**2.-self.theta_lgstd*2.
                             +torch.exp(self.theta_lgstd*2))/2.
    
    def basis_linear(self, x):
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        basis = x.matmul(theta)
        concat_basis = []
        for act in self.act_set:
            if(act == 'gsk'):
                concat_basis.append(torch.cat([torch.sum(torch.stack(basis.cos().chunk(2, -1)), 0), 
                                               torch.sum(torch.stack(basis.sin().chunk(2, -1)), 0)], -1))
            elif(act == 'sin' or act == 'cos'):
                concat_basis.append(getattr(torch, act)(basis))
            else:
                concat_basis.append(getattr(F, act)(basis))
        return torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.theta_mean.size(1)*len(self.act_set))
        
    def forward(self, emb, hidden):
        hx, cx = hidden
        outputs = []
        for i, x in enumerate(emb):
            if(self.uncertain_position == 0):
                x = self.basis_linear(x)
            elif(self.uncertain_position == 1):
                hx = self.basis_linear(hx)
            gates = self.input_map(x)+self.hidden_map(hx)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, -1)
            if(self.uncertain_position == 2):
                ingate = self.basis_linear(ingate)
            else:
                ingate = self.ingate_act(ingate)
            if(self.uncertain_position == 3):
                forgetgate = self.basis_linear(forgetgate)
            else:
                forgetgate = self.forgetgate_act(forgetgate)
            if(self.uncertain_position == 4):
                cellgate = self.basis_linear(cellgate)
            else:
                cellgate = self.cellgate_act(cellgate)
            if(self.uncertain_position == 5):
                outgate = self.basis_linear(outgate)
            else:
                outgate = self.outgate_act(outgate)
            cy = (forgetgate * cx) + (ingate * cellgate)
            if(self.uncertain_position == 6):
                hy = outgate * self.basis_linear(cy)
            else:
                hy = outgate * self.cell_act(cy)
            outputs.append(hy)
            hx, cx = hy, cy
        outputs = torch.cat(outputs, 0)
        return outputs, (hx, cx)

class FastGPLSTM(nn.Module):

    def __init__(self, lstm, hidden_size):
        super(FastGPLSTM, self).__init__()
        self.lstm = lstm
        self.act_set = {'sigmoid', 'tanh', 'relu', 'gsk'}
        self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.init_parameters()
    
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.theta_mean.size(1))
        self.theta_mean.data.uniform_(-stdv, stdv)
        self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return
    
    def kl_divergence(self):
        return torch.sum(self.theta_mean**2.-self.theta_lgstd*2.
                         +torch.exp(self.theta_lgstd*2))/2.

    def forward(self, input, hidden):
        hx, cx = hidden
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        hx_basis = hx.matmul(theta)
        concat_basis = []
        for act in self.act_set:
            if(act == 'gsk'):
                concat_basis.append(torch.cat([torch.sum(torch.stack(basis.cos().chunk(2, -1)), 0), 
                                               torch.sum(torch.stack(basis.sin().chunk(2, -1)), 0)], -1))
            elif(act == 'sin' or act == 'cos'):
                concat_basis.append(getattr(torch, act)(hx_basis))
            else:
                concat_basis.append(getattr(F, act)(hx_basis))
        hx_basis = torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.theta_mean.size(1)*len(self.act_set))
        return self.lstm.forward(input, (hx_basis, cx))

class GRU(nn.Module):

    def __init__(self, input_size, hidden_size, wdrop=0.):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.wdrop = wdrop
        self.updategate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.resetgate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.cellgate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.updategate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.resetgate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.cellgate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.updategate_b = nn.Parameter(torch.rand(hidden_size))
        self.resetgate_b = nn.Parameter(torch.rand(hidden_size))
        self.cellgate_b = nn.Parameter(torch.rand(hidden_size))
        self.init_parameters()

    def init_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)
        
    def sample_dropout_mask(self):
        if(self.training):
            self.updategate_mask = torch.bernoulli(self.updategate_ih.data.new(
                self.updategate_ih.data.size()).fill_(self.wdrop))/self.wdrop
            self.resetgate_mask = torch.bernoulli(self.resetgate_ih.data.new(
                self.resetgate_ih.data.size()).fill_(self.wdrop))/self.wdrop
            self.cellgate_mask = torch.bernoulli(self.cellgate_ih.data.new(
                self.cellgate_ih.data.size()).fill_(self.wdrop))/self.wdrop
        else:
            self.updategate_mask = self.resetgate_mask = self.cellgate_mask = 1.
    
    def forward(self, emb, hidden):
        hx, _ = hidden
        outputs = []
        if(self.wdrop > 0.):
            self.sample_dropout_mask()
        for i, x in enumerate(emb):
            updategate = F.sigmoid(F.linear(x, self.updategate_ih, None)
                    +F.linear(hx[0], self.updategate_hh*self.updategate_mask, self.updategate_b))
            resetgate = F.sigmoid(F.linear(x, self.resetgate_ih, None)
                    +F.linear(hx[0], self.resetgate_hh*self.resetgate_mask, self.resetgate_b))
            cellgate = F.tanh(F.linear(x, self.cellgate_ih, None)
                    +F.linear(resetgate*hx[0], self.cellgate_hh*self.cellgate_mask, self.cellgate_b))
            hy = (1-updategate)*hx+updategate*cellgate
            outputs.append(hy)
            hx = hy
        outputs = torch.stack(outputs, 0)
        return outputs, (hx, hx)

class BayesGRU(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1, wdrop=0.):
        super(BayesGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.uncertain_position = uncertain_position
        self.wdrop = wdrop
        self.updategate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.resetgate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.cellgate_ih = nn.Parameter(torch.rand(hidden_size, input_size))
        self.updategate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.resetgate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.cellgate_hh = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.updategate_b = nn.Parameter(torch.rand(hidden_size))
        self.resetgate_b = nn.Parameter(torch.rand(hidden_size))
        self.cellgate_b = nn.Parameter(torch.rand(hidden_size))
        if(1 <= self.uncertain_position <= 3):
            self.theta_ih_lgstd = nn.Parameter(torch.rand(hidden_size, input_size))
            self.theta_hh_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.init_parameters()
        
    def init_parameters(self):
        stdv = 1. / math.sqrt(self.hidden_size)
        for p in self.parameters():
            p.data.uniform_(-stdv, stdv)
        if(1 <= self.uncertain_position <= 3):
            stdv = 1. / math.sqrt(self.hidden_size)
            self.theta_ih_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
            self.theta_hh_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        if(1 <= self.uncertain_position <= 3):
            if(self.uncertain_position == 1):
                theta_mean = torch.cat([self.updategate_ih, self.updategate_hh], -1)
            elif(self.uncertain_position == 2):
                theta_mean = torch.cat([self.resetgate_ih, self.resetgate_hh], -1)
            elif(self.uncertain_position == 3):
                theta_mean = torch.cat([self.cellgate_ih, self.cellgate_hh], -1)
            theta_lgstd = torch.cat([self.theta_ih_lgstd, self.theta_hh_lgstd], -1)
            return torch.mean(theta_mean**2.-theta_lgstd*2.+torch.exp(theta_lgstd*2))/2.
        return 0

    def sample_weight_diff(self):
        if(self.training):
            theta_ih_std = torch.exp(self.theta_ih_lgstd)
            epsilon = theta_ih_std.new_zeros(*theta_ih_std.size()).normal_()
            theta_ih_diff = epsilon*theta_ih_std
            theta_hh_std = torch.exp(self.theta_hh_lgstd)
            epsilon = theta_hh_std.new_zeros(*theta_hh_std.size()).normal_()
            theta_hh_diff = epsilon*theta_hh_std
            return theta_ih_diff, theta_hh_diff
        return 0, 0

    def forward(self, emb, hx):
        outputs = []
        for i, x in enumerate(emb):
            theta_ih_diff, theta_hh_diff = self.sample_weight_diff()
            updategate = torch.sigmoid(F.linear(x, self.updategate_ih+(theta_ih_diff if self.uncertain_position==1 else 0), None)
                    +F.linear(hx[0], self.updategate_hh+(theta_hh_diff if self.uncertain_position==1 else 0), self.updategate_b))
            resetgate = torch.sigmoid(F.linear(x, self.resetgate_ih+(theta_ih_diff if self.uncertain_position==2 else 0), None)
                    +F.linear(hx[0], self.resetgate_hh+(theta_hh_diff if self.uncertain_position==2 else 0), self.resetgate_b))
            cellgate = torch.tanh(F.linear(x, self.cellgate_ih+(theta_ih_diff if self.uncertain_position==3 else 0), None)
                    +F.linear(resetgate*hx[0], self.cellgate_hh+(theta_hh_diff if self.uncertain_position==3 else 0), self.cellgate_b))
            hy = (1-updategate)*hx+updategate*cellgate
            outputs.append(hy)
            hx = hy
        outputs = torch.cat(outputs, 0)
        outputs = torch.squeeze(outputs, 1)
        return outputs, hx

class FastBayesGRU(nn.Module):

    def __init__(self, input_size, hidden_size, wdrop=0.):
        super(FastBayesGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, num_layers=1, dropout=0.)
        self.wdrop = wdrop
        self._setup()

    def widget_demagnetizer_y2k_edition(*args, **kwargs):
        # We need to replace flatten_parameters with a nothing function
        # It must be a function rather than a lambda as otherwise pickling explodes
        # We can't write boring code though, so ... WIDGET DEMAGNETIZER Y2K EDITION!
        return

    def _setup(self):
        # Terrible temporary solution to an issue regarding compacting weights re: CUDNN RNN
        #if issubclass(type(self.gru), torch.nn.RNNBase):
        #    self.gru.flatten_parameters = self.widget_demagnetizer_y2k_edition

        w_hh = getattr(self.gru, 'weight_hh_l0')
        #del self.gru._parameters['weight_hh_l0']
        self.w_hh_mean = nn.Parameter(w_hh.data[:self.hidden_size])
        self.w_hh_lgstd = nn.Parameter(w_hh.data[:self.hidden_size])
        stdv = 1. / math.sqrt(self.hidden_size)
        self.w_hh_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))

        w_ih = getattr(self.gru, 'weight_ih_l0')
        #del self.gru._parameters['weight_hh_l0']
        self.w_ih_mean = nn.Parameter(w_ih.data[:self.hidden_size])
        self.w_ih_lgstd = nn.Parameter(w_ih.data[:self.hidden_size])
        stdv = 1. / math.sqrt(self.hidden_size)
        self.w_ih_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))

    def _setweights(self):
        if self.training:
            w_hh_std = torch.exp(self.w_hh_lgstd)
            epsilon = w_hh_std.new_zeros(*w_hh_std.size()).normal_()
            w_hh_update = self.w_hh_mean + epsilon*w_hh_std
        else:
            w_hh_update = self.w_hh_mean
        getattr(self.gru, 'weight_hh_l0').data[:self.hidden_size] = w_hh_update

        if self.training:
            w_ih_std = torch.exp(self.w_ih_lgstd)
            epsilon = w_ih_std.new_zeros(*w_ih_std.size()).normal_()
            w_ih_update = self.w_ih_mean + epsilon*w_ih_std
        else:
            w_ih_update = self.w_ih_mean
        getattr(self.gru, 'weight_ih_l0').data[:self.hidden_size] = w_ih_update

    def forward(self, *args):
        self._setweights()
        return self.gru.forward(*args)
    
    def kl_divergence(self):
        return torch.mean(self.w_hh_mean**2.-self.w_hh_lgstd*2.+torch.exp(self.w_hh_lgstd*2))/2.+\
                    torch.mean(self.w_ih_mean**2.-self.w_ih_lgstd*2.+torch.exp(self.w_ih_lgstd*2))/2.

class GPGRU(nn.Module):

    def __init__(self, input_size, hidden_size, uncertain_position=1):
        super(GPGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.act_set = {"sigmoid", "tanh", "relu", "sin", "cos"}
        self.uncertain_position = uncertain_position
        self.updategate_act = nn.Sigmoid()
        self.resetgate_act = nn.Sigmoid()
        self.cellgate_act = nn.Tanh()
        self.lamb = nn.Parameter(torch.rand(len(self.act_set), hidden_size))
        self.updategate_map = nn.Linear(input_size+hidden_size, hidden_size)
        self.resetgate_map = nn.Linear(input_size+hidden_size, hidden_size)
        self.cellgate_map = nn.Linear(input_size+hidden_size, hidden_size)
        if(1 <= self.uncertain_position <= 3):
            self.theta_mean = nn.Parameter(torch.rand(input_size+hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(input_size+hidden_size, hidden_size))
        elif(uncertain_position == 4):
            self.theta_mean = nn.Parameter(torch.rand(input_size, input_size))
            self.theta_lgstd = nn.Parameter(torch.rand(input_size, input_size))
        elif(uncertain_position == 5):
            self.theta_mean = nn.Parameter(torch.rand(hidden_size, hidden_size))
            self.theta_lgstd = nn.Parameter(torch.rand(hidden_size, hidden_size))
        self.init_parameters()
        
    def init_parameters(self):
        if(1 <= self.uncertain_position <= 5):
            stdv = 1. / math.sqrt(self.theta_mean.size(1))
            self.lamb.data.uniform_(-stdv, stdv)
            self.theta_mean.data.uniform_(-stdv, stdv)
            self.theta_lgstd.data.uniform_(2*np.log(stdv), np.log(stdv))
    
    def kl_divergence(self):
        if(1 <= self.uncertain_position <= 5):
            return torch.mean(self.theta_mean**2.-self.theta_lgstd*2.+torch.exp(self.theta_lgstd*2))/2.
        return 0

    def basis_linear(self, x):
        if(self.training):
            theta_std = torch.exp(self.theta_lgstd)
            epsilon = theta_std.new_zeros(*theta_std.size()).normal_()
            theta = self.theta_mean+epsilon*theta_std
        else:
            theta = self.theta_mean
        basis = x.matmul(theta)
        lamb = F.softmax(self.lamb, 1)
        lamb = F.softmax(lamb, 0)
        concat_basis = []
        for i, act in enumerate(self.act_set):
            if(act == 'sin' or act == 'cos'):
                concat_basis.append(lamb[i]*getattr(torch, act)(basis))
            else:
                concat_basis.append(lamb[i]*getattr(F, act)(basis))
        return torch.sum(torch.stack(concat_basis), 0)/math.sqrt(self.theta_mean.size(1))
        
    def forward(self, emb, hidden):
        hx, _ = hidden
        # hx = hidden
        outputs = []
        for i, x in enumerate(emb):
            if(self.uncertain_position == 4):
                x = self.basis_linear(x)
            if(self.uncertain_position == 5):
                _hx = self.basis_linear(hx[0])
                x_hx = torch.cat([x, _hx], -1)
            else:
                x_hx = torch.cat([x, hx[0]], -1)
            if(self.uncertain_position == 1):
                updategate = self.basis_linear(x_hx)
            else:
                updategate = self.updategate_act(self.updategate_map(x_hx))
            if(self.uncertain_position == 2):
                resetgate = self.basis_linear(x_hx)
            else:
                resetgate = self.resetgate_act(self.resetgate_map(x_hx))
            x_rhx = torch.cat([x, resetgate*hx[0]], -1)
            if(self.uncertain_position == 3):
                cellgate = self.basis_linear(x_rhx)
            else:
                cellgate = self.cellgate_act(self.cellgate_map(x_rhx))
            hy = (1-updategate) * hx + (updategate * cellgate)
            outputs.append(hy)
            hx = hy
        outputs = torch.stack(outputs, 0)
        return outputs, (hx, hx)

