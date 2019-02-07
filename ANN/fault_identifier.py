'''
define a fault identifier
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.distributions.normal import Normal
from torch.autograd import Variable

class fault_identifier(nn.Module):
    def __init__(self, hs0_size, x_size, mode_size, state_size, para_size, rnn_size, fc0_size=[], fc1_size=[], fc2_size=[], fc3_size=[], dropout=0.5):
        '''
        Args:
            hs0_size: an int, the size of hybrid states, [m0, m1..., s0, s1...].
            x_size: an int, the size of inputs to the neural network.
            mode_size: a list, [m0, m1, m2..], m_i is the size of each mode variable.
            state_size: an int.
            para_size: an int, the number of fault parameters.
            rnn_size: a list, [hidden_size, num_layers], the inputs size is the x_size
            fc_i_size: a list, the size of the i_th fc module.
                If fc is empty, the corresponding module will be designed from in to out directly.
                If fc is not empty, the module will add extral layers.
        '''
        super(fault_identifier, self).__init__()
        self.hs0_size = hs0_size
        self.x_size = x_size
        self.mode_size = mode_size
        self.state_size = state_size
        self.para_size = para_size
        self.rnn_size = rnn_size
        self.fc0_size = fc0_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        hidden_size, num_layers = rnn_size
        # RNN module, which is the core component.
        self.rnn = nn.GRU(input_size=x_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) # (batch, seq, feature).
        # FC0 module, which converts the hs0 into initial inner states.
        fc0_maps = [hs0_size] + fc0_size + [hidden_size]
        self.fc0 = nn.ModuleList([nn.Linear(fc0_maps[i], fc0_maps[i+1]) for i in range(len(fc0_maps)-1)])
        self.ac0 = nn.ModuleList([nn.PReLU() for _ in range(len(fc0_maps)-1)])
        # FC1 module, which converts the inner states into system modes.
        self.fc1 = nn.ModuleList()
        self.ac1 = nn.ModuleList()
        self.sm1 = nn.ModuleList()
        for m_size in mode_size:
            self.sm1.append(nn.Softmax(dim=2))
            fc1_maps = [hidden_size] + fc1_size + [m_size]
            fc1_tmp = nn.ModuleList([nn.Linear(fc1_maps[i], fc1_maps[i+1]) for i in range(len(fc1_maps)-1)]) # priori
            ac1_tmp = nn.ModuleList([nn.ReLU() for _ in range(len(fc1_maps)-1)]) # priori
            self.fc1.append(fc1_tmp)
            self.ac1.append(ac1_tmp)
        # FC2 module, which converts the inner states into system states.
        fc2_maps = [hidden_size]+ fc2_size + [state_size]
        self.fc21 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # mu
        self.ac21 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-1)])
        self.fc22 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # sigma
        self.ac22 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-1)])
        # FC3 module, which converts the inner states into system parameters.
        fc3_maps = [hidden_size]+ fc3_size + [para_size]
        self.fc31 = nn.ModuleList([nn.Linear(fc3_maps[i], fc3_maps[i+1]) for i in range(len(fc3_maps)-1)]) # mu
        self.ac31 = nn.ModuleList([nn.ReLU() for _ in range(len(fc3_maps)-1)])
        self.fc32 = nn.ModuleList([nn.Linear(fc3_maps[i], fc3_maps[i+1]) for i in range(len(fc3_maps)-1)]) # sigma
        self.ac32 = nn.ModuleList([nn.ReLU() for _ in range(len(fc3_maps)-1)])

    def forward(self, x):
        '''
        x: a tuple, (hs0, x).
        now, x: (batch, seq, feature). 
        '''
        hs0, x = x
        # h0
        h0 = hs0.repeat(self.rnn_size[1], 1, 1)
        for l, a in zip(self.fc0, self.ac0):
            h0 = l(h0)
            h0 = a(h0)
        # RNN/GRU
        hidden_states, _ = self.rnn(x, h0)
        # Modes
        modes = []
        for m, ma, sm in zip(self.fc1, self.ac1, self.sm1):
            mode_m = hidden_states
            for l, a in zip(m, ma):
                mode_m = l(mode_m)
                mode_m = a(mode_m)
            mode_m = sm(mode_m)
            modes.append(mode_m)
        # States
        states_mu = hidden_states
        for l, a in zip(self.fc21, self.ac21):
            states_mu = l(states_mu)
            states_mu = a(states_mu)
        states_sigma = hidden_states
        for l, a in zip(self.fc22, self.ac22):
            states_sigma = l(states_sigma)
            states_sigma = a(states_sigma)
        states_sigma = torch.exp(states_sigma)
        # Paras
        paras_mu = hidden_states
        for l, a in zip(self.fc31, self.ac31):
            paras_mu = l(paras_mu)
            paras_mu = a(paras_mu)
        paras_mu = torch.exp(-paras_mu) # exp(-x), then paras_mu is between 0 and 1
        paras_sigma = hidden_states
        for l, a in zip(self.fc32, self.ac32):
            paras_sigma = l(paras_sigma)
            paras_sigma = a(paras_sigma)
        paras_sigma = torch.exp(-paras_sigma) # exp(-x), then paras_sigma is between 0 and 1
        # modes, states and parameters
        return modes, (states_mu, states_sigma), (paras_mu, paras_sigma) 

def one_mode_cross_entropy(y_head, y):
    '''
    args:
        y_head: batch × mode_size
        y: batch × mode_size
    '''
    ce = - y * torch.log(y_head)
    ce = torch.sum(ce)
    batch, t, _ = y_head.size()
    ce  = ce / (batch*t)
    return ce

def multi_mode_cross_entropy(y_head, y):
    '''
    args:
        y_head: the prediceted values
        y: the real values
    '''
    ce = torch.tensor(0, dtype=torch.float)
    for y1, y0 in zip(y_head, y):
        ce += one_mode_cross_entropy(y1, y0)
    return ce

def normal_stochastic_loss(mu, sigma, obs):
    batch, t, _ = mu.size()
    m = Normal(mu, sigma)
    sample = m.rsample()
    loss = torch.sum((sample-obs)**2 / (sigma**2)) / (batch*t)
    return loss

def np2tensor(x):
    return torch.tensor(x, dtype=torch.float)
