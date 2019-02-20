'''
The identifier of RO systems. Compared with RO_identifier.py, the outputs are different.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,parentdir)
import torch
import argparse
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from fault_identifier import gru_fault_identifier
from fault_identifier import cnn_fault_identifier
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import one_mode_cross_entropy
from utilities.utilities import multi_mode_cross_entropy
from utilities.utilities import normal_stochastic_loss
from utilities.utilities import np2tensor

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, window, limit, normal_proportion, snr_or_pro, mask, output_names):
    r = data_mana.sample(size=batch, window=window, limit=limit, normal_proportion=0.2, \
                         snr_or_pro=snr_or_pro,\
                         norm_o=np.array([1,1,1,10e9,10e8]), \
                         norm_s=np.array([1,1,1,30,10e9,10e8]),\
                         mask=mask,
                         output_names=output_names)
    return r

def show_loss(i, loss, mode_loss, para_loss, state_value_loss, para_value_loss, running_loss):
    running_loss[:] += np.array([loss.item(), mode_loss.item(), para_loss.item(), state_value_loss.item(), para_value_loss.item()])
    if i%10==9:
        ave_loss = running_loss /  10
        msg = '# %d loss:%.3f=%.3f+%.3f+%.3f+%.3f' \
              %(i + 1, ave_loss[0], ave_loss[1], ave_loss[2], ave_loss[3], ave_loss[4])
        print(msg)
        running_loss[:] = np.zeros(5)
    else:
        print('#', end='', flush=True)

def train(epoch, batch, window, limit, data_mana, f_identifier, optimizer, obs_snr, mask, para_mask, output_names):
    train_loss = []
    running_loss = np.zeros(5)
    for i in range(epoch):
        optimizer.zero_grad()

        hs0, x, m, y, p = sample_data(data_mana, batch, window, limit, normal_proportion=0.2, snr_or_pro=obs_snr, mask=mask, output_names=output_names)
        modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)  = f_identifier((np2tensor(hs0), np2tensor(x)))
        
        mode_loss = multi_mode_cross_entropy(modes, data_mana.np2target(m))
        para_loss = one_mode_cross_entropy(paras, data_mana.np2paratarget(p))
        state_value_loss = normal_stochastic_loss(states_mu, states_sigma, np2tensor(y), 10)
        para_value_loss = normal_stochastic_loss(paras_mu, paras_sigma, np2tensor(p), 10, para_mask)
        loss = mode_loss + para_loss + state_value_loss + para_value_loss

        train_loss.append(loss.item())
        show_loss(i, loss, mode_loss, para_loss, state_value_loss, para_value_loss, running_loss)

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
    if torch.cuda.is_available():
        f_identifier.cuda()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ann', type=str, choices=['cnn', 'gru'], help='choose the cnn structure.')
    parser.add_argument('-d', '--data', type=str, help='choose the key values.')
    parser.add_argument('-o', '--output', type=int, help='choose output names.')
    parser.add_argument('-s', '--start', type=int, help='start limit.')
    parser.add_argument('-e', '--end', type=int, help='end limit.')
    args = parser.parse_args()
    window = 5

    ann = args.ann
    data_set = args.data
    output_names = ['q_fp', 'p_tr', 'p_memb', 'e_Cbrine', 'e_Ck'] if args.output==1 else None

    save_path =  os.path.join(this_path, 'RO\\{}'.format(data_set))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    mask = ['f_m']
    para_mask = [0, 0, 0]
    model_name = 'ro2.{}'.format(ann) if args.output==1 else 'ro.{}'.format(ann)
    epoch = 2000
    batch = 500
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\{}\\RO.cfg'.format(data_set))
    if not os.path.exists(data_cfg):
        raise RuntimeError('Data set does not exist.')
    data_mana = new_data_manager(data_cfg, si)
    T = int(window / si)
    limit = (-3 if args.start is None else args.start, 2 if args.end is None else args.end)
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
    train_loss = train(epoch, batch, window, limit, data_mana, f_identifier, optimizer, obs_snr, mask, para_mask, output_names)
    # save model
    save_model(f_identifier, save_path, model_name)
    # figure
    plot(train_loss, save_path, model_name)
