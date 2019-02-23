import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gru_mode_detector(nn.Module):
    def __init__(self, x_size, mode_size, rnn_size, fc_size=[], dropout=0.5):
        '''
        Args:
            x_size: an int, the size of inputs to the neural network.
            mode_size: an int.
            rnn_size: a list, [hidden_size, num_layers], the inputs size is the x_size
            fc_size: a list, the size of the i_th fc module.
                If fc is empty, the corresponding module will be designed from in to out directly.
                If fc is not empty, the module will add extral layers.
        '''
        super(gru_mode_detector, self).__init__()
        self.x_size = x_size
        self.mode_size = mode_size
        self.rnn_size = rnn_size
        self.fc_size = fc_size
        self.hidden_size = rnn_size[0]
        self.num_layers = rnn_size[1]
        hidden_size, num_layers = rnn_size
        # RNN module, which is the core component.
        self.rnn = nn.GRU(input_size=x_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system mode distribution.
        fc_maps = [hidden_size] + fc_size + [mode_size]
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)]) # priori
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-2)]) # Active function for the last linear layer is not here.
        self.sm = nn.Softmax(dim=2)
    def forward(self, x):
        '''
        x: a tuple, (hs0, x).
        now, x: (batch, seq, feature). 
        '''
        # RNN/GRU
        mode, _ = self.rnn(x)
        # Modes
        for l, a in zip(self.fc, self.ac):
            mode = l(mode)
            mode = a(mode)
        mode = self.fc[-1](mode)
        return mode

    def predict_mode(self, x):
        mode = self.forward(x)
        mode = self.sm(mode)
        return mode


class gru_para_fault_isolator(nn.Module):
    def __init__(self, x_size, para_size, rnn_size, fc_size=[], dropout=0.5):
        '''
        Args:
            x_size: an int, the size of inputs to the neural network.
            para_size: an int, the number of fault parameters.
            rnn_size: a list, [hidden_size, num_layers], the inputs size is the x_size
            fc_size: a list, the size of the i_th fc module.
                If fc is empty, the corresponding module will be designed from in to out directly.
                If fc is not empty, the module will add extral layers.
        '''
        super(gru_para_fault_isolator, self).__init__()
        self.x_size = x_size
        self.para_size = para_size
        self.rnn_size = rnn_size
        self.fc_size = fc_size
        hidden_size, num_layers = rnn_size
        # RNN module, which is the core component.
        self.rnn = nn.GRU(input_size=x_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system parameter distribution.
        fc_maps = [hidden_size]+ fc_size + [para_size+1] # no para fault + para fault size
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)])
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-2)])
        self.sm = nn.Softmax(dim=2)

    def forward(self, x):
        # RNN/GRU
        paras, _ = self.rnn(x) # not the real hidden_states, but the output of GRU
        # Para Faults
        for l, a in zip(self.fc, self.ac):
            paras = l(paras)
            paras = a(paras)
        paras = self.fc[-1](paras)
        return paras

    def predict_para_fault(self, x):
        paras = self.forward(x)
        paras = self.sm(paras)
        return paras

class gru_para_fault_identifier(nn.Module):
    def __init__(self, x_size, rnn_size, fc_size=[], dropout=0.5):
        '''
        Args:
            x_size: an int, the size of inputs to the neural network.
            rnn_size: a list, [hidden_size, num_layers], the inputs size is the x_size
            fc_size: a list, the size of the i_th fc module.
                If fc is empty, the corresponding module will be designed from in to out directly.
                If fc is not empty, the module will add extral layers.
        '''
        super(gru_para_fault_identifier, self).__init__()
        self.x_size = x_size
        self.rnn_size = rnn_size
        self.fc_size = fc_size
        hidden_size, num_layers = rnn_size
        # RNN module, which is the core component.
        self.rnn = nn.GRU(input_size=x_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC module, which converts the inner states into system modes.
        fc_maps = [hidden_size]+ fc_size + [1]
        self.fc = nn.ModuleList([nn.Linear(fc_maps[i], fc_maps[i+1]) for i in range(len(fc_maps)-1)])
        self.ac = nn.ModuleList([nn.PReLU() for _ in range(len(fc_maps)-2)])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # RNN/GRU
        paras, _ = self.rnn(x)
        # Fault Para Values
        for l, a in zip(self.fc, self.ac):
            paras = l(paras)
            paras = a(paras)
        paras = self.fc[-1](paras)
        paras = self.sigmoid(paras)
        return paras
