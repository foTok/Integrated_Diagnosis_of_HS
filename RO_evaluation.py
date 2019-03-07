'''
Test the PF by RO System
'''
import os
import torch
import argparse
import logging
import numpy as np
import matplotlib as mpl
from utilities import chi2_confidence
from utilities import exp_confidence
from integrated_particle_fiter import ipf
from data_manager import data_manager
from RO import RO
from utilities import obtain_var

mpl.rc('font',family='Times New Roman')
# get parameters from environment
parser = argparse.ArgumentParser()
parser.add_argument('-t', '--test', type=str, help='test set name')
parser.add_argument('-o', '--output', type=str, help='output directory')
parser.add_argument('-r', '--repeat', type=int, help='repeat times')
parser.add_argument('-n', '--num', type=int, help='particle number')
parser.add_argument('-s', '--start', type=int, help='start index')
parser.add_argument('-m', '--mode', type=str, help='filter mode')
parser.add_argument('-c', '--conf', type=float, help='obs confidence')
args = parser.parse_args()

# settings
si = 0.01
process_snr = 45
obs_snr = 20
obs_scale =np.array([1, 1, 1, 10, 10e8])
repeat = 10 if args.repeat is None else args.repeat
N = 50 if args.num is None else args.num
test = 'test' if args.test is None else args.test
start = 0 if args.start is None else args.start
filter_mode = 'ann' if args.mode is None else args.mode
mode_detector = 'model/mode_detector'
pf_isolator = 'model/pf_isolator'
f_f_identifier = 'model/f_f_identifier'
f_r_identifier = 'model/f_r_identifier'
f_m_identifier = 'model/f_m_identifier'
data_cfg = '{}/RO.cfg'.format(test)
data_mana = data_manager(data_cfg, si)
# log directory
log_path = 'log/RO/{}'.format(args.output)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

for k in range(repeat):
    for i in range(start, len(data_mana.data)):
        msg = '\n************Track the {}th observation, {}th time.************'.format(i, k)
        print(msg)
        logging.info(msg)
        # figure path
        fig_path = os.path.join(log_path, str(i))
        if not os.path.isdir(fig_path):
            os.makedirs(fig_path)
        # prepare evaluation environment.
        fault_type, fault_magnitude, fault_time, msg = data_mana.get_info(i, prt=False)
        ref_mode = data_mana.select_modes(i)
        ref_state = data_mana.select_states(i)
        output = data_mana.select_outputs(i)
        output_with_noise = data_mana.select_outputs(i, obs_snr)
        state_sigma = np.sqrt(obtain_var(ref_state, process_snr))
        state_sigma[-1] = 1
        obs_sigma = np.sqrt(obtain_var(output, obs_snr))
        # create tracker and start tracking.
        with torch.no_grad():
            ro = RO(si)
            tracker = ipf(ro, state_sigma, obs_sigma)
            tracker.set_obs_conf(args.conf)
            tracker.set_filter_mode(filter_mode)
            tracker.load_mode_detector(mode_detector)
            tracker.load_pf_isolator(pf_isolator)
            tracker.load_identifier([f_f_identifier, f_r_identifier, f_m_identifier])
            tracker.set_scale(obs_scale)
            tracker.log_msg(msg)
            tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=N)
            # tracker.track(mode=0, state_mu=np.array([ 1.35286541e-02,  9.98226267e-01,  8.08528738e-01,  5.28445410e-02,  -4.73411158e+03,  6.90904974e+02]), state_sigma=np.zeros(6), obs=output_with_noise[9900:15000], N=N)
            tracker.plot_state(file_name=os.path.join(fig_path, 'states{}'.format(k)))
            tracker.plot_mode(file_name=os.path.join(fig_path, 'modes{}'.format(k)))
            tracker.plot_res(file_name=os.path.join(fig_path, 'res{}'.format(k)))
            tracker.plot_para(file_name=os.path.join(fig_path, 'paras{}'.format(k)))
            tracker.plot_Z(file_name=os.path.join(fig_path, 'Z{}'.format(k)))
            tracker.evaluate_modes(ref_mode)
            tracker.evaluate_states(ref_state)
            # tracker.plot_Z()
            # tracker.plot_res()
            # tracker.plot_para_fault()
