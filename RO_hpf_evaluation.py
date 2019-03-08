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
    si = 0.01
    process_snr = 45
    obs_snr = 20
    test = 'test' if args.test is None else args.test
    N = 150 if args.num is None else args.num

    data_cfg = '{}/RO.cfg'.format(test)
    data_mana = data_manager(data_cfg, si)
    log_path = 'log/RO/{}'.format(args.output)
    if not os.path.isdir(log_path):
        os.makedirs(log_path)
    logging.basicConfig(filename=os.path.join(log_path, 'log.txt'), level=logging.INFO)

    for k in range(args.repeat):
        for i in range(len(data_mana.data)):
            msg = data_mana.get_info(i, prt=False)
            state = data_mana.select_states(i)
            state_with_noise = data_mana.select_states(i, process_snr)
            output = data_mana.select_outputs(i)
            output_with_noise = data_mana.select_outputs(i, obs_snr)
            modes = data_mana.select_modes(i)

            ref_mode = data_mana.select_modes(i)
            ref_mode = ref_mode%3
            ref_state = data_mana.select_states(i)

            state_sigma = np.sqrt(obtain_var(state, process_snr))
            state_sigma[-1] = 1
            obs_sigma = np.sqrt(obtain_var(output, obs_snr))

            fig_path = os.path.join(log_path, str(i))
            if not os.path.isdir(fig_path):
                os.makedirs(fig_path)

            ro = RO(si)
            tracker = hpf(ro, state_sigma, obs_sigma)
            tracker.log_msg(msg)
            tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=N)
            tracker.plot_state(file_name=os.path.join(fig_path, 'states{}'.format(k)))
            tracker.plot_mode(file_name=os.path.join(fig_path, 'modes{}'.format(k)))
            tracker.evaluate_modes(ref_mode)
            tracker.evaluate_states(ref_state)
