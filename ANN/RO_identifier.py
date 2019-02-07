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
from fault_identifier import fault_identifier
from fault_identifier import multi_mode_cross_entropy
from fault_identifier import normal_stochastic_loss
from fault_identifier import np2tensor
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, window, limit, normal_proportion, snr_or_pro):
    r = data_mana.sample(size=20, window=5, limit=(2,2), normal_proportion=0.2, \
                         snr_or_pro=snr_or_pro,\
                         norm_o=np.array([1,1,1,10e9,10e8]), \
                         norm_s=np.array([1,1,1,30,10e9,10e8]))
    return r

def show_loss(i, loss, mode_loss, state_loss, para_loss, running_loss):
    running_loss[:] += np.array([loss.item(), mode_loss.item(), state_loss.item(), para_loss.item()])
    if i%10==9:
        ave_loss = running_loss/10
        msg = '%d loss:%.5f=%.5f+%.5f+%.5f' %(i + 1, ave_loss[0], ave_loss[1], ave_loss[2], ave_loss[3])
        print(msg)
        running_loss[:] = np.zeros(4)

def train(epoch, batch, data_mana, f_identifier, optimizer, obs_snr):
    train_loss = []
    running_loss = np.zeros(4)
    for i in range(epoch):
        optimizer.zero_grad()

        hs0, x, m, y, p = sample_data(data_mana, batch, window=5, limit=(1,2), normal_proportion=0.2, snr_or_pro=obs_snr)
        modes, (states_mu, states_sigma), (paras_mu, paras_sigma)  = f_identifier((np2tensor(hs0), np2tensor(x)))
        
        mode_loss = multi_mode_cross_entropy(modes, data_mana.np2target(m))
        state_loss = normal_stochastic_loss(states_mu, states_sigma, np2tensor(y))
        para_loss = normal_stochastic_loss(paras_mu, paras_sigma, np2tensor(p))
        loss = mode_loss + state_loss + para_loss

        train_loss.append(loss.item())
        show_loss(i, loss, mode_loss, state_loss, para_loss, running_loss)

        loss.backward()
        optimizer.step()
    return train_loss

def save_model(model, path, name):
    torch.save(f_identifier, os.path.join(path, name))
    print('saved model {} to {}'.format(name, path))

def plot(train_loss, path, name):
    plt.cla()
    plt.plot(np.array(train_loss))
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(path, name+'.svg'), format='svg')



if __name__ == "__main__":
    debug = False
    key = 'debug' if debug else 'train'
    save_path =  os.path.join(this_path, 'RO\\{}'.format(key))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    model_name = 'ro_{}'.format(key)
    epoch = 2000
    batch = 1000
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\{}\\RO.cfg'.format(key))
    data_mana = new_data_manager(data_cfg, si)
    # the model
    f_identifier = fault_identifier(hs0_size=7,\
                        x_size=5,\
                        mode_size=[6],\
                        state_size=6,\
                        para_size=3,\
                        rnn_size=[32, 8],\
                        fc0_size=[128, 64, 64, 32],\
                        fc1_size=[128, 64, 64, 32],\
                        fc2_size=[128, 64, 64, 32],\
                        fc3_size=[128, 64, 64, 32])
    # optimizer
    optimizer = optim.Adam(f_identifier.parameters(), lr=0.001, weight_decay=8e-3)
    # train
    train_loss = train(epoch, batch, data_mana, f_identifier, optimizer, obs_snr)
    # save model
    save_model(f_identifier, save_path, model_name)
    # figure
    plot(train_loss, save_path, model_name)
