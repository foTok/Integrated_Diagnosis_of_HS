'''
define a fault identifier
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class gru_fault_identifier(nn.Module):
    def __init__(self, hs0_size, x_size, mode_size, state_size, para_size, rnn_size, fc0_size=[], fc1_size=[], fc2_size=[], fc3_size=[], fc4_size=[], dropout=0.5):
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
        super(gru_fault_identifier, self).__init__()
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
        self.fc0 = nn.ModuleList()
        self.ac0 = nn.ModuleList()
        for _ in range(rnn_size[1]):
            fc0_tmp = nn.ModuleList([nn.Linear(fc0_maps[i], fc0_maps[i+1]) for i in range(len(fc0_maps)-1)])
            ac0_tmp = nn.ModuleList([nn.PReLU() for _ in range(len(fc0_maps)-1)])
            self.fc0.append(fc0_tmp)
            self.ac0.append(ac0_tmp)
        # FC1 module, which converts the inner states into system modes.
        self.fc1 = nn.ModuleList()
        self.ac1 = nn.ModuleList()
        self.sm1 = nn.ModuleList()
        for m_size in mode_size:
            fc1_maps = [hidden_size] + fc1_size + [m_size]
            fc1_tmp = nn.ModuleList([nn.Linear(fc1_maps[i], fc1_maps[i+1]) for i in range(len(fc1_maps)-1)]) # priori
            ac1_tmp = nn.ModuleList([nn.PReLU() for _ in range(len(fc1_maps)-2)]) # Activation function for the last linear layer is not here.
            self.fc1.append(fc1_tmp)
            self.ac1.append(ac1_tmp)
            self.sm1.append(nn.Softmax(dim=2))
        # FC2 module, which converts the inner states into system states.
        fc2_maps = [hidden_size]+ fc2_size + [state_size]
        self.fc21 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # mu
        self.ac21 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-1)])
        self.fc22 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # sigma
        self.ac22 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-2)])
        self.sigmoid22 = nn.Sigmoid()
        # FC3 module, which converts the inner states into system parameter faults
        fc3_maps = [hidden_size]+ fc3_size + [para_size+1] # no para fault + para fault size
        self.fc3 = nn.ModuleList([nn.Linear(fc3_maps[i], fc3_maps[i+1]) for i in range(len(fc3_maps)-1)])
        self.ac3 = nn.ModuleList([nn.PReLU() for _ in range(len(fc3_maps)-2)])
        self.sm3 = nn.Softmax(dim=2)
        # FC4 module, which converts the inner states into system parameters.
        fc4_maps = [hidden_size]+ fc4_size + [para_size]
        self.fc41 = nn.ModuleList([nn.Linear(fc4_maps[i], fc4_maps[i+1]) for i in range(len(fc4_maps)-1)]) # mu
        self.ac41 = nn.ModuleList([nn.PReLU() for _ in range(len(fc4_maps)-2)])
        self.sigmoid41 = nn.Sigmoid()
        self.fc42 = nn.ModuleList([nn.Linear(fc4_maps[i], fc4_maps[i+1]) for i in range(len(fc4_maps)-1)]) # sigma
        self.ac42 = nn.ModuleList([nn.PReLU() for _ in range(len(fc4_maps)-2)])
        self.sigmoid42 = nn.Sigmoid()

    def forward(self, x):
        '''
        x: a tuple, (hs0, x).
        now, x: (batch, seq, feature). 
        '''
        hs0, x = x
        # h0
        inner_h0 = []
        for fc0, ac0 in zip(self.fc0, self.ac0):
            h0 = hs0
            for l, a in zip(fc0, ac0):
                h0 = l(h0)
                h0 = a(h0)
            t, h = h0.size()
            h0 = h0.view(1, t, h)
            inner_h0.append(h0)
        inner_h0 = torch.cat(inner_h0)
        # RNN/GRU
        hidden_states, _ = self.rnn(x, inner_h0)
        # Modes
        modes = []
        for m, ma, sm in zip(self.fc1, self.ac1, self.sm1):
            mode_m = hidden_states
            for l, a in zip(m, ma):
                mode_m = l(mode_m)
                mode_m = a(mode_m)
            mode_m = m[-1](mode_m)
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
        states_sigma = self.fc22[-1](states_sigma)
        states_sigma = self.sigmoid22(states_sigma)
        # Para Faults
        paras = hidden_states
        for l, a in zip(self.fc3, self.ac3):
            paras = l(paras)
            paras = a(paras)
        paras = self.fc3[-1](paras)
        paras = self.sm3(paras)
        # Fault Para Values
        paras_mu = hidden_states
        for l, a in zip(self.fc41, self.ac41):
            paras_mu = l(paras_mu)
            paras_mu = a(paras_mu)
        paras_mu = self.fc41[-1](paras_mu)
        paras_mu = self.sigmoid41(paras_mu)
        paras_sigma = hidden_states
        for l, a in zip(self.fc42, self.ac42):
            paras_sigma = l(paras_sigma)
            paras_sigma = a(paras_sigma)
        paras_sigma = self.fc42[-1](paras_sigma)
        paras_sigma = self.sigmoid42(paras_sigma)
        # modes, states and parameters
        last_modes = [m[:,-1,:] for m in modes]
        last_paras = paras[:,-1,:]
        last_states_mu = states_mu[:,-1,:]
        last_states_sigma = states_sigma[:,-1,:] 
        last_paras_mu = paras_mu[:,-1,:]
        last_paras_sigma = paras_sigma[:,-1,:]
        return last_modes, last_paras, (last_states_mu, last_states_sigma), (last_paras_mu, last_paras_sigma)

class cnn_fault_identifier(nn.Module):
    def __init__(self, hs0_size, x_size, mode_size, state_size, para_size, cnn_size, fc0_size=[], fc1_size=[], fc2_size=[], fc3_size=[], fc4_size=[], T=500):
        '''
        cnn_size: (channel number,  kernel number)
        '''
        super(cnn_fault_identifier, self).__init__()
        self.hs0_size = hs0_size
        self.x_size = x_size
        self.mode_size = mode_size
        self.state_size = state_size
        self.para_size = para_size
        self.cnn_size = cnn_size
        self.fc0_size = fc0_size
        self.fc1_size = fc1_size
        self.fc2_size = fc2_size
        self.fc3_size = fc3_size
        self.T = T
        self.pooling = 2
        # RNN module, which is the core component.
        self.cnn = nn.ModuleList()
        self.cnn_ac = nn.ModuleList()
        self.cnn_pool = nn.ModuleList()
        channels = [x_size] + cnn_size[0]
        kernels = cnn_size[1]
        Lout = T
        for k in kernels:
            Lout = (Lout - k + 1) # CNN Layer
            Lout = int(Lout/self.pooling) # pool
        for i in range(len(kernels)):
            self.cnn.append(nn.Conv1d(channels[i], channels[i+1], kernels[i]))
            self.cnn_ac.append(nn.PReLU())
            self.cnn_pool.append(nn.AvgPool1d(self.pooling))
        self.cnn_merge = nn.Conv1d(channels[-1], channels[-1], Lout)
        # FC0 module, which converts the hs0 into initial features.
        fc0_maps = [hs0_size] + fc0_size
        self.fc0 = nn.ModuleList([nn.Linear(fc0_maps[i], fc0_maps[i+1]) for i in range(len(fc0_maps)-1)])
        self.ac0 = nn.ModuleList([nn.PReLU() for _ in range(len(fc0_maps)-1)])
        feature_num = channels[-1] + fc0_size[-1]
        # FC1 module, which converts the inner states into system modes.
        self.fc1 = nn.ModuleList()
        self.ac1 = nn.ModuleList()
        self.sm1 = nn.ModuleList()
        for m_size in mode_size:
            fc1_maps = [feature_num] + fc1_size + [m_size]
            fc1_tmp = nn.ModuleList([nn.Linear(fc1_maps[i], fc1_maps[i+1]) for i in range(len(fc1_maps)-1)]) # priori
            ac1_tmp = nn.ModuleList([nn.PReLU() for _ in range(len(fc1_maps)-2)]) # Activation function for the last linear layer is not here.
            self.fc1.append(fc1_tmp)
            self.ac1.append(ac1_tmp)
            self.sm1.append(nn.Softmax(dim=1))
        # FC2 module, which converts the inner states into system states.
        fc2_maps = [feature_num]+ fc2_size + [state_size]
        self.fc21 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # mu
        self.ac21 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-1)])
        self.fc22 = nn.ModuleList([nn.Linear(fc2_maps[i], fc2_maps[i+1]) for i in range(len(fc2_maps)-1)]) # sigma
        self.ac22 = nn.ModuleList([nn.PReLU() for _ in range(len(fc2_maps)-2)])
        self.sigmoid22 = nn.Sigmoid()
        # FC3 module, which converts the inner states into system parameter faults
        fc3_maps = [feature_num]+ fc3_size + [para_size+1] # no para fault + para fault size
        self.fc3 = nn.ModuleList([nn.Linear(fc3_maps[i], fc3_maps[i+1]) for i in range(len(fc3_maps)-1)])
        self.ac3 = nn.ModuleList([nn.PReLU() for _ in range(len(fc1_maps)-2)])
        self.sm3 = nn.Softmax(dim=1)
        # FC4 module, which converts the inner states into system parameters.
        fc4_maps = [feature_num]+ fc4_size + [para_size]
        self.fc41 = nn.ModuleList([nn.Linear(fc4_maps[i], fc4_maps[i+1]) for i in range(len(fc4_maps)-1)]) # mu
        self.ac41 = nn.ModuleList([nn.PReLU() for _ in range(len(fc4_maps)-2)])
        self.sigmoid41 = nn.Sigmoid()
        self.fc42 = nn.ModuleList([nn.Linear(fc4_maps[i], fc4_maps[i+1]) for i in range(len(fc4_maps)-1)]) # sigma
        self.ac42 = nn.ModuleList([nn.PReLU() for _ in range(len(fc4_maps)-2)])
        self.sigmoid42 = nn.Sigmoid()

    def forward(self, x):
        '''
        x: a tuple, (hs0, x).
        now, x: (batch, seq, feature). 
        '''
        hs0, x = x
        x = x.permute(0, 2, 1) # now, x: (batch, feature, seq). 
        # h0 => init_features: batch × fc0_size[-1]
        init_features = hs0
        for l, a in zip(self.fc0, self.ac0):
            init_features = l(init_features)
            init_features = a(init_features)
        # CNN
        cnn_features = x
        for c, a, p in zip(self.cnn, self.cnn_ac, self.cnn_pool):
            cnn_features = c(cnn_features)
            cnn_features = a(cnn_features)
            cnn_features = p(cnn_features)
        cnn_features = self.cnn_merge(cnn_features)
        cnn_features = cnn_features.view(-1, self.cnn_size[0][-1])
        # merge init_features and cnn_features
        features = torch.cat((init_features, cnn_features), 1)
        # Modes
        modes = []
        for m, ma, sm in zip(self.fc1, self.ac1, self.sm1):
            mode_m = features
            for l, a in zip(m, ma):
                mode_m = l(mode_m)
                mode_m = a(mode_m)
            mode_m = m[-1](mode_m)
            mode_m = sm(mode_m)
            modes.append(mode_m)
        # States
        states_mu = features
        for l, a in zip(self.fc21, self.ac21):
            states_mu = l(states_mu)
            states_mu = a(states_mu)
        states_sigma = features
        for l, a in zip(self.fc22, self.ac22):
            states_sigma = l(states_sigma)
            states_sigma = a(states_sigma)
        states_sigma = self.fc22[-1](states_sigma)
        states_sigma = self.sigmoid22(states_sigma)
        # Para Faults
        paras = features
        for l, a in zip(self.fc3, self.ac3):
            paras = l(paras)
            paras = a(paras)
        paras = self.fc3[-1](paras)
        paras = self.sm3(paras)
        # Fault Para Values
        paras_mu = features
        for l, a in zip(self.fc41, self.ac41):
            paras_mu = l(paras_mu)
            paras_mu = a(paras_mu)
        paras_mu = self.fc41[-1](paras_mu)
        paras_mu = self.sigmoid41(paras_mu)
        paras_sigma = features
        for l, a in zip(self.fc42, self.ac42):
            paras_sigma = l(paras_sigma)
            paras_sigma = a(paras_sigma)
        paras_sigma = self.fc42[-1](paras_sigma)
        paras_sigma = self.sigmoid42(paras_sigma)
        # modes, states and parameters
        return modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)
