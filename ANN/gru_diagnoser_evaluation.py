import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var
from utilities.utilities import np2tensor

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
parser.add_argument('-t', '--type', type=str, help='model type.')
parser.add_argument('-o', '--output', type=str, help='output directory')
args = parser.parse_args()

model_name = 'ro.gru' if args.model_name is None else args.model_name
this_path = os.path.dirname(os.path.abspath(__file__))
log_path = 'eval' if args.output is None else 'eval\\{}'.format(args.output)
log_path = 'RO\\{}'.format(log_path)
log_path = os.path.join(this_path, log_path)

# set log path
if not os.path.exists(log_path):
    os.makedirs(log_path)

use_cuda = False
si = 0.01
# test data set
obs_snr = 20
state_scale =np.array([1, 1, 1, 30, 10e9, 10e8])
obs_scale =np.array([1, 1, 1, 10e9, 10e8])
data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\test_d\\RO.cfg')
data_mana = data_manager(data_cfg, si)
# diagnoser
diagnoser = os.path.join(parentdir, 'ANN\\model\\{}'.format(model_name))
diagnoser = torch.load(diagnoser, map_location='cpu')
diagnoser.eval()

# set log
logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

# ro
ro = RO(si)

msg = 'evaluate model {}.'.format(model_name)
log_msg(msg)

for i in range(len(data_mana.data)):
    msg = 'evaluating data {}.'.format(i)
    log_msg(msg)
    _, _, _, msg = data_mana.get_info(i, prt=False)
    log_msg(msg)

    output_with_noise = data_mana.select_outputs(i, obs_snr)
    time, obs = output_with_noise.shape
    output_with_noise = output_with_noise.reshape(1, time, obs)
    output_with_noise = output_with_noise / obs_scale
    output_with_noise = np2tensor(output_with_noise, use_cuda)


    ref_mode = data_mana.select_modes(i)
    ref_state = data_mana.select_states(i)

    y_head = diagnoser(output_with_noise)
    _, time, mode_dis = y_head.size()
    y_head = y_head.detach().numpy().reshape(-1, mode_dis)
    
    if args.type=='detector':
        mode = np.argmax(y_head, axis=1)
        ro.plot_modes(mode, file_name=os.path.join(log_path, 'modes_d{}'.format(i)))
    elif args.type=='isolator':
        x = np.arange(len(y_head))*si
        plt.plot(x, y_head)
        plt.legend(['n', 'f_f', 'f_r', 'f_m'])
        plt.show()
        plt.close()

        fp = np.argmax(y_head, axis=1)
        plt.plot(x, fp)
        plt.savefig(os.path.join(log_path, 'fp{}.svg'.format(i)), format='svg')
        plt.close()

