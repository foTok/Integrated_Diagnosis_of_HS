'''
Test the ANN.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import torch
import numpy as np
import matplotlib.pyplot as plt
from fault_identifier import fault_identifier
from fault_identifier import multi_mode_cross_entropy
from fault_identifier import normal_stochastic_loss
from fault_identifier import np2tensor
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO

if __name__ == '__main__':
    si = 0.01
    obs_snr = 20
    data_cfg = parentdir + '\\Systems\\RO_System\\data\\debug\\RO.cfg'
    data_mana = data_manager(data_cfg, si)
    
    hs0, x, m, y, p = data_mana.sample(size=20, window=5, limit=(2,2), normal_proportion=0.2, \
                                       norm_o=np.array([1,1,1,10e9,10e8]), \
                                       norm_s=np.array([1,1,1,30,10e9,10e8]))

    ft = fault_identifier(hs0_size=7,\
                          x_size=5,\
                          mode_size=[6],\
                          state_size=6,\
                          para_size=3,\
                          rnn_size=[8, 2],\
                          fc0_size=[32, 32],\
                          fc1_size=[32, 32],\
                          fc2_size=[32, 32],\
                          fc3_size=[32, 32])

    modes, (states_mu, states_sigma), (paras_mu, paras_sigma)  = ft((np2tensor(hs0), np2tensor(x)))

    mode_loss = multi_mode_cross_entropy(modes, data_mana.np2target(m))

    state_loss = normal_stochastic_loss(states_mu, states_sigma, np2tensor(y))

    para_loss = normal_stochastic_loss(paras_mu, paras_sigma, np2tensor(p))

    loss = mode_loss + state_loss + para_loss

    loss.backward()

    print('End of File.')
