import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gru_fault_diagnoser(nn.Module):
    def __init__(self, x_size, mode_size, state_size, para_size, rnn_size, fc1_size=[], fc2_size=[], fc3_size=[], dropout=0.5):
        '''
        Args:
            x_size: an int, the size of inputs to the neural network.
            mode_size: an int.
            state_size: an int.
            para_size: an int, the number of fault parameters.
            rnn_size: a list, [hidden_size, num_layers], the inputs size is the x_size
            fc_i_size: a list, the size of the i_th fc module.
                If fc is empty, the corresponding module will be designed from in to out directly.
                If fc is not empty, the module will add extral layers.
        '''
        super(gru_fault_diagnoser, self).__init__()
        self.x_size = x_size
        self.mode_size = mode_size
        self.state_size = state_size
        self.para_size = para_size
        self.rnn_size = rnn_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.hidden_size = rnn_size[0]
        self.num_layers = rnn_size[1]
        hidden_size, num_layers = rnn_size
        # RNN module, which is the core component.
        self.rnn = nn.GRU(input_size=x_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC1 module, which converts the inner states into system modes.
        fc1_maps = [hidden_size] + fc1_size + [mode_size]
        self.fc1 = nn.ModuleList([nn.Linear(fc1_maps[i], fc1_maps[i+1]) for i in range(len(fc1_maps)-1)]) # priori
        self.ac1 = nn.ModuleList([nn.PReLU() for _ in range(len(fc1_maps)-2)]) # Activation function for the last linear layer is not here.
        self.sm1 = nn.Softmax(dim=2)
        # FC2 module, which converts the inner states into system states.
        fc2_maps = [hidden_size]+ fc2_size + [state_size]
        self.fc2 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # mu
        self.ac2 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-1)])
        # FC3 module, which converts the inner states into system parameter faults
        fc3_maps = [hidden_size]+ fc3_size + [para_size]
        self.fc3 = nn.ModuleList([nn.Linear(fc3_maps[i], fc3_maps[i+1]) for i in range(len(fc3_maps)-1)]) # mu
        self.ac3 = nn.ModuleList([nn.PReLU() for _ in range(len(fc3_maps)-2)])
        self.sigmoid3 = nn.Sigmoid()

    def forward(self, x):
        '''
        x: a tuple, (hs0, x).
        now, x: (batch, seq, feature). 
        '''
        batch, _, _ = x.size()
        inner_h0 = torch.zeros(self.num_layers, batch, self.hidden_size).cuda() if torch.cuda.is_available() else torch.zeros(self.num_layers, batch, self.hidden_size).cuda()
        # RNN/GRU
        hidden_states, _ = self.rnn(x, inner_h0) # the hidden states here is the outputs of GRU, not real hidden states.
        # Modes
        mode = hidden_states
        for l, a in zip(self.fc1, self.ac1):
            mode = l(mode)
            mode = a(mode)
        mode = self.fc1[-1](mode)
        mode = self.sm1(mode)
        # States
        state = hidden_states
        for l, a in zip(self.fc2, self.ac2):
            state = l(state)
            state = a(state)
        # Fault Para Values
        para = hidden_states
        for l, a in zip(self.fc3, self.ac3):
            para = l(para)
            para = a(para)
        para = self.fc3[-1](para)
        para = self.sigmoid3(para)

        return mode, state, para