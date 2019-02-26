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
from torch.nn import CrossEntropyLoss
from gru_diagnoser import gru_mode_detector
from gru_diagnoser import gru_para_fault_isolator
from gru_diagnoser import gru_para_fault_identifier
from lstm_diagnoser import lstm_mode_detector
from lstm_diagnoser import lstm_para_fault_isolator
from lstm_diagnoser import lstm_para_fault_identifier
from cnn_gru_diagnoser import cnn_gru_mode_detector
from cnn_gru_diagnoser import cnn_gru_pf_isolator
from cnn_gru_diagnoser import cnn_gru_pf_identifier
from data_manager import data_manager
from utilities import np2tensor

def mse(input, target, use_cuda=True):
    '''
    input: tensor, should be cuda is avaliable
    target: np.array
    '''
    target = torch.tensor(target).float().cuda() if torch.cuda.is_available()  and use_cuda \
             else torch.tensor(target).float()
    loss = MSELoss()
    return loss(input, target)

def cross_entropy(input, target, use_cuda=True):
    '''
    input: tensor, should be cuda is avaliable
    target: np.array
    '''
    target = torch.tensor(target).long().cuda() if torch.cuda.is_available() and use_cuda \
             else torch.tensor(target).long()
    target = target.view(-1)
    _, _, C = input.size()
    input = input.view(-1, C)  
    loss = CrossEntropyLoss()
    return loss(input, target)

def new_data_manager(cfg, si):
    data_mana = data_manager(cfg, si)
    return data_mana

def sample_data(data_mana, batch, normal_proportion, snr_or_pro, mask, obs, res):
    r = data_mana.sample_all(size=batch, \
                    normal_proportion=normal_proportion, \
                    snr_or_pro=snr_or_pro,\
                    norm_o=np.array([1,1,1,10,10e9]), \
                    norm_s=np.array([1,1,1,10,10e9,10e8]),\
                    mask=mask,\
                    output_names=obs,\
                    res=res)
    return r

def show_loss(i, loss, running_loss):
    running_loss += loss.item()
    if i%10==9:
        ave_loss = running_loss /  10
        msg = '# %d loss:%.3f' %(i + 1, ave_loss)
        print(msg)
        running_loss = 0
    else:
        print('#', end='', flush=True)
    return running_loss

def train(save_path, model_name, model_type, epoch, batch, normal_proportion, \
       data_mana, diagnoser, optimizer, obs_snr, mask, obs, use_cuda, res):
    train_loss = []
    running_loss = 0

    for i in range(epoch):
        optimizer.zero_grad()

        x, m, fp_mode, fp_value = sample_data(data_mana, batch, normal_proportion=normal_proportion, snr_or_pro=obs_snr, mask=mask, obs=obs, res=res)
        m = m%3
        y_head = diagnoser(np2tensor(x, use_cuda))

        if model_type=='detector':
            loss = cross_entropy(y_head, m, use_cuda)
        elif model_type=='isolator':
            loss = cross_entropy(y_head, fp_mode, use_cuda)
        elif model_type=='f_f' or model_type=='f_r':
            fp_vec = (np.sum(fp_value, (0, 1))!=0)
            index = np.sum(fp_vec*np.array([0, 1, 2]))
            y = fp_value[:, :, [index]]
            loss = 400*mse(y_head, y, use_cuda)
        else:
            raise RuntimeError('Unknown Type.')

        train_loss.append(loss.item())
        running_loss = show_loss(i, loss, running_loss)

        loss.backward()
        optimizer.step()
        save_model(diagnoser, save_path, model_name)
        plot(train_loss, save_path, model_name)
    return train_loss

def get_gru_model(t, use_cuda=True):
    print('GRU Model')
    if t=='detector':
        model = gru_mode_detector(x_size=5,\
                          mode_size=3,\
                          rnn_size=[64, 4],\
                          fc_size=[128, 64, 32])
    elif t=='isolator':
        model = gru_para_fault_isolator(x_size=5,\
                            para_size=3,\
                            rnn_size=[64, 4],\
                            fc_size=[128, 64, 32])
    elif t=='identifier':
        model = gru_para_fault_identifier(x_size=5,\
                              rnn_size=[64, 4],\
                              fc_size=[128, 64, 32])
    else:
        raise RuntimeError('Unknown Type.')
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    return model

def get_lstm_model(t, use_cuda=True):
    print('LSTM Model')
    if t=='detector':
        model = lstm_mode_detector(x_size=5,\
                          mode_size=3,\
                          rnn_size=[64, 4],\
                          fc_size=[128, 64, 32])
    elif t=='isolator':
        model = lstm_para_fault_isolator(x_size=5,\
                            para_size=3,\
                            rnn_size=[64, 4],\
                            fc_size=[128, 64, 32])
    elif t=='identifier':
        model = lstm_para_fault_identifier(x_size=5,\
                              rnn_size=[64, 4],\
                              fc_size=[128, 64, 32])
    else:
        raise RuntimeError('Unknown Type.')
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    return model

def get_cnn_gru_model(t, use_cuda=True):
    print('CNN-GRU Model')
    if t=='detector':
        model = cnn_gru_mode_detector( x_size=5,\
                           cnn_feature_map=[32, 64, 128, 64], cnn_kernel_size=[64, 32, 16, 8],\
                           num_layers=2, hidden_size=64, dropout=0.5, \
                           fc_size=[64, 32], mode_size=3)
    elif t=='isolator':
        model = cnn_gru_pf_isolator(x_size=5,\
                        cnn_feature_map=[32, 64, 128, 64], cnn_kernel_size=[64, 32, 16, 8],\
                        num_layers=2, hidden_size=64, dropout=0.5, \
                        fc_size=[64, 32], pf_size=3)
    elif t=='identifier':
        model = cnn_gru_pf_identifier( x_size=5,\
                           cnn_feature_map=[32, 64, 128, 64], cnn_kernel_size=[64, 32, 16, 8],\
                           num_layers=2, hidden_size=64, dropout=0.5, \
                           fc_size=[64, 32])
    else:
        raise RuntimeError('Unknown Type.')
    if torch.cuda.is_available() and use_cuda:
        model.cuda()
    return model

def get_model(net, t, use_cuda):
    if net=='gru':
        return get_gru_model(t, use_cuda)
    elif net=='lstm':
        return get_lstm_model(t, use_cuda)
    elif net=='cnn_gru':
        return get_cnn_gru_model(t, use_cuda)
    else:
        raise RuntimeError('Unknown Net Type.')

def save_model(model, path, name):
    torch.save(model, os.path.join(path, name))

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
    parser.add_argument('-t', '--type', type=str, help='model type.')
    parser.add_argument('-n', '--net', type=str, help='network type.')
    parser.add_argument('-r', '--res', type=str, help='if sample residual.')
    parser.add_argument('-o', '--output', type=str, help='choose output names.')
    args = parser.parse_args()

    use_cuda = True
    obs = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine']
    res = False if args.res is None else True

    data_set = args.data
    # mask
    if args.type=='detector':# mode detector
        model = get_model(args.net, 'detector', use_cuda)
        mask = ['f_m']
        normal_proportion = 0.1
    elif args.type=='isolator': # fault parameter isolator
        model = get_model(args.net, 'isolator', use_cuda)
        mask = ['s_normal', 's_pressure', 's_reverse', 'f_m']
        normal_proportion = 0.05
    elif args.type=='f_f':
        model = get_model(args.net, 'identifier', use_cuda)
        mask = ['normal', 's_normal', 's_pressure', 's_reverse', 'f_r', 'f_m']
        normal_proportion = 0
    elif args.type=='f_r':
        model = get_model(args.net, 'identifier', use_cuda)
        mask = ['normal', 's_normal', 's_pressure', 's_reverse', 'f_f', 'f_m']
        normal_proportion = 0
    else:
        raise RuntimeError('Unknown Type.')
    model_name = args.output

    save_path =  'model'
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    
    epoch = 4000
    batch = 7
    # data manager
    si = 0.01
    obs_snr = 20
    data_cfg = '{}/RO.cfg'.format(data_set)
    if not os.path.exists(data_cfg):
        raise RuntimeError('Data set does not exist.')
    data_mana = new_data_manager(data_cfg, si)
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-3)
    # train
    train(save_path, model_name, args.type, epoch, batch, normal_proportion, \
        data_mana, model, optimizer, obs_snr, mask, obs, use_cuda, res)
