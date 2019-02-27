'''
Test the PF by RO System
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import argparse
import logging
import matplotlib as mpl
from particle_fiter import chi2_confidence
from particle_fiter import exp_confidence
from particle_fiter import hs_system_wrapper
from particle_fiter import hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var

mpl.rc('font',family='Times New Roman')
# get parameters from environment
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--conf', type=str, choices=['exp', 'chi2'], help='confidence')
parser.add_argument('-t', '--test', type=str, help='test set name')
parser.add_argument('-o', '--output', type=str, help='output directory')
parser.add_argument('-r', '--repeat', type=int, help='repeat times')
parser.add_argument('-n', '--num', type=int, help='particle number')
args = parser.parse_args()

# settings
si = 0.01
process_snr = 45
obs_snr = 20
obs_scale =np.array([1, 1, 1, 10, 10e9])
repeat = 10 if args.repeat is None else args.repeat
conf = chi2_confidence if args.conf=='chi2' else exp_confidence
N = 50 if args.num is None else args.num
test = 'test' if args.test is None else args.test
mode_detector = os.path.join(parentdir, 'ANN/model/cnn_gru_mode_detector')
pf_isolator = os.path.join(parentdir, 'ANN/model/cnn_gru_pf_isolator_res')
f_f_identifier = os.path.join(parentdir, 'ANN/model/cnn_gru_f_f_identifier_res')
f_r_identifier = os.path.join(parentdir, 'ANN/model/cnn_gru_f_r_identifier_res')
data_cfg = os.path.join(parentdir, 'Systems/RO_System/{}/RO.cfg'.format(test))
data_mana = data_manager(data_cfg, si)
# log directory
log_path = 'log\\RO\\{}'.format(args.output)
if not os.path.isdir(log_path):
    os.makedirs(log_path)
logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

for k in range(repeat):
    for i in range(len(data_mana.data)):
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
        ro = RO(si)
        hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
        tracker = hpf(hsw, conf=conf)
        tracker.load_mode_detector(mode_detector)
        tracker.load_pf_isolator(pf_isolator)
        tracker.load_identifier([f_f_identifier, f_r_identifier])
        tracker.set_scale(obs_scale)
        tracker.log_msg(msg)
        tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=N)
        tracker.plot_state(file_name=os.path.join(fig_path, 'states{}'.format(k)))
        tracker.plot_mode(file_name=os.path.join(fig_path, 'modes{}'.format(k)))
        tracker.plot_res(file_name=os.path.join(fig_path, 'res{}'.format(k)))
        tracker.plot_para(file_name=os.path.join(fig_path, 'paras{}'.format(k)))
        tracker.evaluate_modes(ref_mode)
        tracker.evaluate_states(ref_state)
