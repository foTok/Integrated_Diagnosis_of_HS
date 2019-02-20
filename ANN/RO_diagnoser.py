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
from torch.nn import MSELoss
from torch.nn import NLLLoss
from gru_diagnoser import gru_fault_diagnoser
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import np2tensor

def mse(input, target):
    '''
    input: tensor, should be cuda is avaliable
    target: np.array
    '''
    target = torch.tensor(target).float().cuda() if torch.cuda.is_available() else torch.tensor(target).float().cuda() 
    loss = MSELoss()
    return loss(input, target)

def cross_entropy(input, target):
    '''
    input: tensor, should be cuda is avaliable
    target: np.array
    '''
    batch,_,_ = input.size()
    loss = NLLLoss()
    ce = torch.tensor(0, dtype=torch.float).cuda() if torch.cuda.is_available() else torch.tensor(0, dtype=torch.float)
    input = torch.log(input)
    for i, t in zip(input, target):
        ce += loss(i, torch.tensor(t).long())/batch
    return ce

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, normal_proportion, snr_or_pro, mask, output_names):
    r = data_mana.sample_all(size=batch, \
                            normal_proportion=0.2, \
                            snr_or_pro=snr_or_pro,\
                            norm_o=np.array([1,1,1,10e9,10e8]), \
                            norm_s=np.array([1,1,1,30,10e9,10e8]),\
                            mask=mask,
                            output_names=output_names)
    return r

def show_loss(i, loss, mode_loss, state_loss, para_loss, running_loss):
    running_loss[:] += np.array([loss.item(), mode_loss.item(), state_loss.item(), para_loss.item()])
    if i%10==9:
        ave_loss = running_loss /  10
        msg = '# %d loss:%.3f=%.3f+%.3f+%.3f' \
              %(i + 1, ave_loss[0], ave_loss[1], ave_loss[2], ave_loss[3])
        print(msg)
        running_loss[:] = 0
    else:
        print('#', end='', flush=True)

def train(save_path, model_name, epoch, batch, data_mana, diagnoser, optimizer, obs_snr, mask, para_mask, output_names):
    train_loss = []
    min_loss = float('inf')
    running_loss = np.zeros(4)

    for i in range(epoch):
        optimizer.zero_grad()

        x, m, y, p = sample_data(data_mana, batch, normal_proportion=0.2, snr_or_pro=obs_snr, mask=mask, output_names=output_names)
        mode, state, para = diagnoser(np2tensor(x))

        mode_loss = cross_entropy(mode, m)
        state_loss = mse(state, y)
        para_loss = mse(para, p)
        loss = mode_loss + state_loss + para_loss

        train_loss.append(loss.item())
        show_loss(i, loss, mode_loss, state_loss, para_loss, running_loss)

        if loss.item() < min_loss:
            save_model(diagnoser, save_path, model_name)

        loss.backward()
        optimizer.step()
    return train_loss

def get_model():
    diagnoser = gru_fault_diagnoser(x_size=5,\
                                    mode_size=6,\
                                    state_size=6,\
                                    para_size=3,\
                                    rnn_size=[32, 4],\
                                    fc1_size=[128, 64, 32],\
                                    fc2_size=[128, 64, 32],\
                                    fc3_size=[128, 64, 32])
    if torch.cuda.is_available():
        diagnoser.cuda()
    return diagnoser

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
    parser.add_argument('-d', '--data', type=str, help='choose the key values.')
    parser.add_argument('-o', '--output', type=int, help='choose output names.')
    args = parser.parse_args()

    data_set = args.data
    output_names = ['q_fp', 'p_tr', 'p_memb', 'e_Cbrine', 'e_Ck'] if args.output==1 else None

    save_path =  os.path.join(this_path, 'RO\\{}'.format(data_set))
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    mask = ['f_m']
    para_mask = [0, 0, 0]
    model_name = 'ro2.gru' if args.output==1 else 'ro.gru'
    epoch = 2000
    batch = 20
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\{}\\RO.cfg'.format(data_set))
    if not os.path.exists(data_cfg):
        raise RuntimeError('Data set does not exist.')
    data_mana = new_data_manager(data_cfg, si)
    # get model
    diagnoser = get_model()
    # optimizer
    optimizer = optim.Adam(diagnoser.parameters(), lr=0.001, weight_decay=1e-2)
    # train
    train_loss = train(save_path, model_name, epoch, batch, data_mana, diagnoser, optimizer, obs_snr, mask, para_mask, output_names)
    # figure
    plot(train_loss, save_path, model_name)
