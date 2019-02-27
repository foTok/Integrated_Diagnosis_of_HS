import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cnn_gru_mode_detector(nn.Module):
    def __init__(self, x_size, \
                 cnn_feature_map, cnn_kernel_size, \
                 num_layers, hidden_size, dropout, \
                 fc_size, mode_size):
        super(cnn_gru_mode_detector, self).__init__()
        self.x_size = x_size
        self.cnn_feature_map = cnn_feature_map
        self.cnn_kernel_size = cnn_kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.mode_size = mode_size
        # CNN
        channels = [x_size] + cnn_feature_map
        self.cnn_padding_layer = nn.ModuleList()
        self.cnn_layer = nn.ModuleList()
        self.cnn_ac_layer = nn.ModuleList()
        for i in range(len(cnn_kernel_size)):
            self.cnn_padding_layer.append(nn.ConstantPad1d((cnn_kernel_size[i]-1, 0), 0))
            self.cnn_layer.append(nn.Conv1d(channels[i], channels[i+1], cnn_kernel_size[i]))
            self.cnn_ac_layer.append(nn.PReLU())
        # RNN
        self.rnn = nn.GRU(input_size=cnn_feature_map[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system mode distribution.
        fc_maps = [hidden_size] + fc_size + [mode_size]
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)])
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-2)]) # Active function for the last linear layer is not here.
        self.sm = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (batch, sequence, channel) => (batch, channel, sequence)
        x = x.permute([0, 2, 1])
        # CNN
        for p, c, a in zip(self.cnn_padding_layer, self.cnn_layer, self.cnn_ac_layer):
            x = p(x)
            x = c(x)
            x = a(x)
        # x: (batch, channel, sequence) => (batch, sequence, channel)
        x = x.permute([0, 2, 1])
        # RNN/GRU
        x, _ = self.rnn(x)
        # FC
        for l, a in zip(self.fc, self.ac):
            x = l(x)
            x = a(x)
        x = self.fc[-1](x)
        return x

    def predict_mode(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class cnn_gru_pf_isolator(nn.Module):
    def __init__(self, x_size, \
                 cnn_feature_map, cnn_kernel_size, \
                 num_layers, hidden_size, dropout, \
                 fc_size, pf_size):
        super(cnn_gru_pf_isolator, self).__init__()
        self.x_size = x_size
        self.cnn_feature_map = cnn_feature_map
        self.cnn_kernel_size = cnn_kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pf_size = pf_size
        # CNN
        channels = [x_size] + cnn_feature_map
        self.cnn_padding_layer = nn.ModuleList()
        self.cnn_layer = nn.ModuleList()
        self.cnn_ac_layer = nn.ModuleList()
        for i in range(len(cnn_kernel_size)):
            self.cnn_padding_layer.append(nn.ConstantPad1d((cnn_kernel_size[i]-1, 0), 0))
            self.cnn_layer.append(nn.Conv1d(channels[i], channels[i+1], cnn_kernel_size[i]))
            self.cnn_ac_layer.append(nn.PReLU())
        # RNN
        self.rnn = nn.GRU(input_size=cnn_feature_map[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system mode distribution.
        fc_maps = [hidden_size] + fc_size + [pf_size+1]
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)])
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-2)]) # Active function for the last linear layer is not here.
        self.sm = nn.Softmax(dim=2)

    def forward(self, x):
        # x: (batch, sequence, channel) => (batch, channel, sequence)
        x = x.permute([0, 2, 1])
        # CNN
        for p, c, a in zip(self.cnn_padding_layer, self.cnn_layer, self.cnn_ac_layer):
            x = p(x)
            x = c(x)
            x = a(x)
        # x: (batch, channel, sequence) => (batch, sequence, channel)
        x = x.permute([0, 2, 1])
        # RNN/GRU
        x, _ = self.rnn(x)
        # FC
        for l, a in zip(self.fc, self.ac):
            x = l(x)
            x = a(x)
        x = self.fc[-1](x)
        return x

    def predict_pf(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

class cnn_gru_pf_identifier(nn.Module):
    def __init__(self, x_size, \
                 cnn_feature_map, cnn_kernel_size, \
                 num_layers, hidden_size, dropout, \
                 fc_size):
        super(cnn_gru_pf_identifier, self).__init__()
        self.x_size = x_size
        self.cnn_feature_map = cnn_feature_map
        self.cnn_kernel_size = cnn_kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        # CNN
        channels = [x_size] + cnn_feature_map
        self.cnn_padding_layer = nn.ModuleList()
        self.cnn_layer = nn.ModuleList()
        self.cnn_ac_layer = nn.ModuleList()
        for i in range(len(cnn_kernel_size)):
            self.cnn_padding_layer.append(nn.ConstantPad1d((cnn_kernel_size[i]-1, 0), 0))
            self.cnn_layer.append(nn.Conv1d(channels[i], channels[i+1], cnn_kernel_size[i]))
            self.cnn_ac_layer.append(nn.PReLU())
        # RNN
        self.rnn = nn.GRU(input_size=cnn_feature_map[-1], hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system mode distribution.
        fc_maps = [hidden_size] + fc_size + [1]
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)])
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-1)]) # Active function for the last linear layer is not here.

    def forward(self, x):
        # x: (batch, sequence, channel) => (batch, channel, sequence)
        x = x.permute([0, 2, 1])
        # CNN
        for p, c, a in zip(self.cnn_padding_layer, self.cnn_layer, self.cnn_ac_layer):
            x = p(x)
            x = c(x)
            x = a(x)
        # x: (batch, channel, sequence) => (batch, sequence, channel)
        x = x.permute([0, 2, 1])
        # RNN/GRU
        x, _ = self.rnn(x)
        # FC
        for l, a in zip(self.fc, self.ac):
            x = l(x)
            x = a(x)
        return x

class ann_step:
    def __init__(self, ann):
        self.ann = ann
        self.obs = []
        self.N = np.sum(ann.cnn_kernel_size) - len(ann.cnn_kernel_size) + 1
        self.hidden = None
        self.out = []

    def step(self, obs):
        self.obs.append(obs)
        if len(self.obs) < self.N:
            x = np.pad(self.obs, ((self.N-len(self.obs), 0), (0, 0)), 'constant')
        else:
            x = np.array(self.obs[-self.N:])
        x = torch.tensor(x, dtype=torch.float)
        seq, cha = x.size()
        x = x.view(1, seq, cha)
        # x: (batch, sequence, channel) => (batch, channel, sequence)
        x = x.permute([0, 2, 1])
        # CNN
        for c, a in zip(self.ann.cnn_layer, self.ann.cnn_ac_layer):
            x = c(x)
            x = a(x)
        # x: (batch, channel, sequence) => (batch, sequence, channel)
        x = x.permute([0, 2, 1])
        # RNN/GRU
        if self.hidden is None:
            x, self.hidden = self.ann.rnn(x)
        else:
            x, self.hidden = self.ann.rnn(x, self.hidden)
        # FC
        for l, a in zip(self.ann.fc, self.ann.ac):
            x = l(x)
            x = a(x)
        if len(self.ann.fc)!=len(self.ann.ac):
            x = self.ann.fc[-1](x)
        x = x[0,0,:].detach().numpy()
        self.out.append(x)
        return x

    def latest_out(self):
        if not self.out:
            return None
        else:
            return self.out[-1]
