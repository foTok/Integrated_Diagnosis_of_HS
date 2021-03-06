import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class cnn_gru_obs_model(nn.Module):
    def __init__(self, x_size, \
                 cnn_feature_map, cnn_kernel_size, \
                 num_layers, hidden_size, dropout, \
                 fc_size, o_size):
        super(cnn_gru_obs_model, self).__init__()
        self.x_size = x_size
        self.cnn_feature_map = cnn_feature_map
        self.cnn_kernel_size = cnn_kernel_size
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.o_size = o_size
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
        fc_maps = [hidden_size] + fc_size + [o_size]
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
