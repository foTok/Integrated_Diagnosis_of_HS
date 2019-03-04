import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from scipy.special import softmax
from data_manager import data_manager
from RO import RO
from cnn_gru_diagnoser import ann_step
from utilities import obtain_var
from utilities import np2tensor

def log_msg(msg):
    logging.info(msg)
    print(msg)

def evaluate_modes(modes, ref_modes):
    if len(modes.shape)==1:
        ref_modes = ref_modes.reshape(-1)
    res = np.abs(modes - ref_modes)
    wrong_num = np.sum(res!=0, 0)
    accuracy = np.mean(res==0, 0)
    msg = '\tWrong mode number = {}, accuracy = {}.'.format(wrong_num, np.round(accuracy, 4))
    log_msg(msg)

def evaluate_states(est_states, ref_states):
    # abs error 
    res = est_states - ref_states
    mu = np.mean(res, 0)
    sigma = np.std(res, 0)
    msg = '\tStates error mu = {}, sigma = {}.'.format(np.round(mu, 4), np.round(sigma, 4))
    log_msg(msg)
    # relative error
    # normalize factor
    ref_mu = np.mean(ref_states, 0)
    ref_sigma = np.std(ref_states, 0)
    # normalize ref states
    n_ref_states = (ref_states - ref_mu)/ref_sigma
    # normalize estimated states
    n_est_states = (est_states - ref_mu)/ref_sigma
    n_res = n_est_states - n_ref_states
    n_mu = np.mean(n_res, 0)
    n_sigma = np.std(n_res, 0)
    msg = '\tStates error n_mu = {}, n_sigma = {}.'.format(np.round(n_mu, 4), np.round(n_sigma, 4))
    log_msg(msg)

# get parameters from environment
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model_name', type=str, help='model name')
parser.add_argument('-d', '--data_set', type=str, help='data_set')
parser.add_argument('-t', '--type', type=str, help='model type.')
parser.add_argument('-r', '--res', type=str, help='if sample residual.')
parser.add_argument('-o', '--output', type=str, help='output directory')
args = parser.parse_args()

res = False if args.res is None else True
model_name = args.model_name
this_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'log/eval/{}'.format(args.output)

# set log path
if not os.path.exists(log_path):
    os.makedirs(log_path)

use_cuda = False
si = 0.01
# test data set
obs_snr = 20
obs_scale =np.array([1, 1, 1, 10, 10e8])
data_cfg = '{}/RO.cfg'.format(args.data_set)
data_mana = data_manager(data_cfg, si)
# diagnoser
diagnoser = 'model/{}'.format(model_name)
diagnoser = torch.load(diagnoser, map_location='cpu')
diagnoser.eval()

# set log
logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

# data 0
data_mana0 = data_manager('test_n\\RO.cfg', si)
output_n = data_mana0.select_outputs(0, norm=obs_scale)
# ro
ro = RO(si)

msg = 'evaluate model {}.'.format(model_name)
log_msg(msg)

for i in range(len(data_mana.data)):
    msg = 'evaluating data {}.'.format(i)
    log_msg(msg)
    _, _, _, msg = data_mana.get_info(i, prt=False)
    log_msg(msg)

    output_with_noise = data_mana.select_outputs(i, obs_snr, norm=obs_scale)
    output_with_noise = output_with_noise - output_n if res else output_with_noise
    # time, obs = output_with_noise.shape
    # output_with_noise = output_with_noise.reshape(1, time, obs)
    # output_with_noise = np2tensor(output_with_noise, use_cuda)


    ref_mode = data_mana.select_modes(i)
    ref_state = data_mana.select_states(i)

    with torch.no_grad():
        # y_head = diagnoser(output_with_noise)
        # _, time, feature = y_head.size()
        # y_head = y_head.detach().numpy().reshape(-1, feature)
        ann_steper = ann_step(diagnoser)
        y_head = []
        for obs in output_with_noise:
            y = ann_steper.step(obs)
            y_head.append(y)
    y_head = np.array(y_head)
    
    if args.type=='detector':
        mode = np.argmax(y_head, axis=1)
        ro.plot_modes(mode, file_name=os.path.join(log_path, 'modes_{}_{}'.format(args.data_set, i)))
    elif args.type=='isolator':
        x = np.arange(len(y_head))*si
        y_head = softmax(y_head, axis=1)
        plt.plot(x, y_head)
        plt.legend(['n', 'f_f', 'f_r', 'f_m'])
        plt.show()
        plt.close()

        fp = np.argmax(y_head, axis=1)
        plt.plot(x, fp)
        plt.savefig(os.path.join(log_path, 'fp{}.svg'.format(i)), format='svg')
        plt.close()
    elif args.type=='f_f':
        x = np.arange(len(y_head))*si
        y_head = y_head.reshape(-1)
        plt.plot(x, y_head)
        plt.show()
        plt.savefig(os.path.join(log_path, 'f_f{}.svg'.format(i)), format='svg')
        plt.close()
    elif args.type=='f_r':
        pass
    else:
        raise RuntimeError('Unknown Type.')
