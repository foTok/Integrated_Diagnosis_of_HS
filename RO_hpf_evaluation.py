'''
Test the PF by RO System
'''
import os
import torch
import numpy as np
import argparse
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from hybrid_particle_filter import hpf
from data_manager import data_manager
from RO import RO
from utilities import obtain_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--repeat', type=int, help='repeat time')
    parser.add_argument('-n', '--num', type=int, help='particle number')
    parser.add_argument('-t', '--test', type=str, help='test_set')
    parser.add_argument('-o', '--output', type=str, help='out directory')
    args = parser.parse_args()
    mpl.rc('font',family='Times New Roman')
    # read parameters from environment
    index = 0
    si = 0.01
    process_snr = 45
    obs_snr = 20
    test = 'test' if args.test is None else args.test
    N = 150 if args.num is None else args.num

    data_cfg = '{}/RO.cfg'.format(test)
    data_mana = data_manager(data_cfg, si)
    msg = data_mana.get_info(index, prt=False)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)
    modes = data_mana.select_modes(index)

    ref_mode = data_mana.select_modes(index)
    ref_state = data_mana.select_states(index)

    state_sigma = np.sqrt(obtain_var(state, process_snr))
    state_sigma[-1] = 1
    obs_sigma = np.sqrt(obtain_var(output, obs_snr))

    log_path = 'log/RO/{}'.format(args.output)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

    for i in range(args.repeat):
        ro = RO(si)
        tracker = hpf(ro, state_sigma, obs_sigma)
        tracker.log_msg(msg)
        tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=N)
        tracker.plot_state(file_name=os.path.join(log_path, 'states{}'.format(i)))
        tracker.plot_mode(file_name=os.path.join(log_path, 'modes{}'.format(i)))
        tracker.evaluate_modes(ref_mode)
        tracker.evaluate_states(ref_state)
