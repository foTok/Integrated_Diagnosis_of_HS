'''
Test the PF by RO System
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import argparse
from particle_fiter import chi2_confidence
from particle_fiter import exp_confidence
from particle_fiter import hs_system_wrapper
from particle_fiter import hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var

# get parameters from environment
parser = argparse.ArgumentParser()
parser.add_argument('-s', '--start', type=int, help='start index')
parser.add_argument('-r', '--repeat', type=int, help='repeat times')
args = parser.parse_args()

start = 0 if args.start is None else args.start
repeat = 10 if args.repeat is None else args.repeat
print('repeat experiments {} times, start from index {}.'.format(repeat, start))

fd, fp = 3, 8
si = 0.01
process_snr = 45
obs_snr = 20
limit = (2, 3)
proportion = 1.0
state_scale =np.array([1, 1, 1, 30, 10e9, 10e8])
obs_scale =np.array([1, 1, 1, 10e9, 10e8])
identifier = os.path.join(parentdir, 'ANN\\RO\\train\\ro.cnn')
data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\test\\RO.cfg')
data_mana = data_manager(data_cfg, si)

for k in range(start, start+repeat):
    for i in range(len(data_mana.data)):
        print('Track the {}th observation, {}th time.'.format(i, k))
        # prepare evaluation environment.
        _, _, _, msg = data_mana.get_info(i, prt=False)
        ref_mode = data_mana.select_modes(i)
        ref_state = data_mana.select_states(i)
        output = data_mana.select_outputs(i)
        output_with_noise = data_mana.select_outputs(i, obs_snr)
        state_sigma = np.sqrt(obtain_var(ref_state, process_snr))
        obs_sigma = np.sqrt(obtain_var(output, obs_snr))
        log_path = 'log/RO/{}'.format(i)
        if not os.path.isdir(log_path):
            os.makedirs(log_path)
        # create tracker and start tracking.
        ro = RO(si)
        hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
        tracker = hpf(hsw)
        tracker.load_identifier(identifier)
        tracker.set_scale(state_scale, obs_scale)
        tracker.set_log('log/RO/{}/log{}.txt'.format(i, k))
        tracker.log_msg(msg)
        tracker.track(modes=0, state_mean=np.zeros(6), state_var=np.zeros(6), \
                    observations=output_with_noise, limit=limit, \
                    fd=fd, fp=fp, proportion=proportion, \
                    Nmin=150, Nmax=200)
        tracker.plot_states(file_name='log/RO/{}/states{}'.format(i, k))
        tracker.plot_modes(file_name='log/RO/{}/modes{}'.format(i, k))
        tracker.plot_res(file_name='log/RO/{}/res{}'.format(i, k))
        tracker.plot_Z(file_name='log/RO/{}/Z{}'.format(i, k))
        tracker.plot_paras(file_name='log/RO/{}/paras{}'.format(i, k))
        tracker.evaluate_modes(ref_mode)
        tracker.evaluate_states(ref_state)
