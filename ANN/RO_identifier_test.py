'''
The identifier of RO systems.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,parentdir)
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from utilities.utilities import one_mode_cross_entropy
from utilities.utilities import multi_mode_cross_entropy
from utilities.utilities import normal_stochastic_loss
from utilities.utilities import np2tensor
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, window, limit, normal_proportion, snr_or_pro):
    r = data_mana.sample(size=batch, window=5, limit=(2,2), normal_proportion=0.2, \
                         snr_or_pro=snr_or_pro,\
                         norm_o=np.array([1,1,1,10e9,10e8]), \
                         norm_s=np.array([1,1,1,30,10e9,10e8]))
    return r

def identify_fault(hs0, x):
    modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)  = f_identifier((np2tensor(hs0), np2tensor(x)))
    return modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)

def load_model(file_name):
    model = torch.load(file_name)
    model.eval()
    return model

def loss(modes, paras, states_mu, states_sigma, paras_mu, paras_sigma, m, p, y):
    mode_loss = multi_mode_cross_entropy(modes, data_mana.np2target(m))
    para_loss = one_mode_cross_entropy(paras, data_mana.np2paratarget(p))
    state_value_loss, _ = normal_stochastic_loss(states_mu, states_sigma, np2tensor(y))
    para_value_loss, _ = normal_stochastic_loss(paras_mu, paras_sigma, np2tensor(p))
    loss = mode_loss + para_loss + state_value_loss + para_value_loss

    msg = 'loss:%.3f=%.3f+%.3f+%.3f+%.3f' %(loss.item(), mode_loss.item(), para_loss.item(), state_value_loss.item(), para_value_loss.item())
    print(msg)

if __name__ == "__main__":
    model_name = os.path.join(parentdir, 'ANN\\RO\\train3\\ro.cnn2')
    epoch = 0
    batch = 100
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\train\\RO.cfg')
    data_mana = new_data_manager(data_cfg, si)
    # the model
    f_identifier = load_model(model_name)
    # sample data
    hs0, x, m, y, p = sample_data(data_mana, batch, window=5, limit=(1,2), normal_proportion=0.2, snr_or_pro=obs_snr)
    # identify fault
    modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma) = identify_fault(hs0, x)
    # compute loss
    loss(modes, paras, states_mu, states_sigma, paras_mu, paras_sigma, m, p, y)
