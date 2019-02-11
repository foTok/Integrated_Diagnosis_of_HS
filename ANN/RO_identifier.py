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
from fault_identifier import gru_fault_identifier
from fault_identifier import cnn_fault_identifier
from fault_identifier import one_mode_cross_entropy
from fault_identifier import multi_mode_cross_entropy
from fault_identifier import normal_stochastic_loss
from fault_identifier import np2tensor
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, window, limit, normal_proportion, snr_or_pro, mask):
    r = data_mana.sample(size=batch, window=window, limit=limit, normal_proportion=0.2, \
                         snr_or_pro=snr_or_pro,\
                         norm_o=np.array([1,1,1,10e9,10e8]), \
                         norm_s=np.array([1,1,1,30,10e9,10e8]),\
                         mask=mask)
    return r

def show_loss(i, loss, mode_loss, para_loss, state_value_loss, para_value_loss, mean_state_value_loss, mean_para_value_loss, \
              running_state_mean_loss, running_para_mean_loss, running_loss):
    running_loss[:] += np.array([loss.item(), mode_loss.item(), para_loss.item(), state_value_loss.item(), para_value_loss.item()])
    running_state_mean_loss[:] += mean_state_value_loss
    running_para_mean_loss[:] += mean_para_value_loss
    if i%10==9:
        ave_loss = running_loss /  10
        ave_state_loss = running_state_mean_loss / 10
        ave_para_loss = running_para_mean_loss / 10
        msg = '# %d loss:%.3f=%.3f+%.3f \
              +%.3f(%.3f+%.3f+%.3f+%.3f+%.3f+%.3f) \
              +%.3f(%.3f+%.3f+%.3f)' \
              %(i + 1, ave_loss[0], ave_loss[1], ave_loss[2], \
              ave_loss[3], ave_state_loss[0], ave_state_loss[1], ave_state_loss[2], ave_state_loss[3], ave_state_loss[4], ave_state_loss[5], \
              ave_loss[4], ave_para_loss[0], ave_para_loss[1], ave_para_loss[2])
        print(msg)
        running_loss[:] = np.zeros(5)
        running_state_mean_loss[:] = np.zeros(6)
        running_para_mean_loss[:] = np.zeros(3)
    else:
        print('#', end='', flush=True)

def train(epoch, batch, window, limit, data_mana, f_identifier, optimizer, obs_snr, mask):
    train_loss = []
    running_loss = np.zeros(5)
    running_state_mean_loss = np.zeros(6)
    running_para_mean_loss = np.zeros(3)
    for i in range(epoch):
        optimizer.zero_grad()

        hs0, x, m, y, p = sample_data(data_mana, batch, window, limit, normal_proportion=0.2, snr_or_pro=obs_snr, mask=mask)
        modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)  = f_identifier((np2tensor(hs0), np2tensor(x)))
        
        mode_loss = multi_mode_cross_entropy(modes, data_mana.np2target(m))
        para_loss = one_mode_cross_entropy(paras, data_mana.np2paratarget(p))
        state_value_loss, mean_state_value_loss = normal_stochastic_loss(states_mu, states_sigma, np2tensor(y), k=10)
        para_value_loss, mean_para_value_loss = normal_stochastic_loss(paras_mu, paras_sigma, np2tensor(p), k=10)
        loss = mode_loss + para_loss + state_value_loss + para_value_loss

        train_loss.append(loss.item())
        show_loss(i, loss, mode_loss, para_loss, state_value_loss, para_value_loss, mean_state_value_loss, mean_para_value_loss, \
                  running_state_mean_loss, running_para_mean_loss, running_loss)

        loss.backward()
        optimizer.step()
    return train_loss

def gru_model():
    f_identifier = gru_fault_identifier(hs0_size=7,\
                    x_size=5,\
                    mode_size=[6],\
                    state_size=6,\
                    para_size=3,\
                    rnn_size=[32, 4],\
                    fc0_size=[64, 32],\
                    fc1_size=[128, 64, 32],\
                    fc2_size=[128, 64, 32],\
                    fc3_size=[128, 64, 32],
                    fc4_size=[128, 64, 32])
    return f_identifier

def cnn_model(T):
    f_identifier = cnn_fault_identifier(hs0_size=7,\
                    x_size=5,\
                    mode_size=[6],\
                    state_size=6,\
                    para_size=3,\
                    cnn_size=([32, 64, 128, 256], [8, 4, 4, 4]),\
                    fc0_size=[64, 32],\
                    fc1_size=[128, 64, 32],\
                    fc2_size=[128, 64, 32],\
                    fc3_size=[128, 64, 32],
                    fc4_size=[128, 64, 32],
                    T=T)
    return f_identifier

def save_model(model, path, name):
    torch.save(model, os.path.join(path, name))
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
    window = 5
    ann = 'cnn' # 'gru
    key = 'debug' if debug else 'train'
    save_path =  os.path.join(this_path, 'RO\\{}'.format(key))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    mask = ['f_m']
    model_name = 'ro0.{}'.format(ann)
    epoch = 2000
    batch = 500
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\{}\\RO.cfg'.format(key))
    data_mana = new_data_manager(data_cfg, si)
    T = int(window / si)
    limit = (1, 2)
    # the model
    if ann=='cnn':
        f_identifier = cnn_model(T)
    elif ann=='gru':
        f_identifier = gru_model()
    else:
        raise RuntimeError('Unknown Model Type.')
    # optimizer
    optimizer = optim.Adam(f_identifier.parameters(), lr=0.001, weight_decay=1e-2)
    # train
    train_loss = train(epoch, batch, window, limit, data_mana, f_identifier, optimizer, obs_snr, mask)
    # save model
    save_model(f_identifier, save_path, model_name)
    # figure
    plot(train_loss, save_path, model_name)
