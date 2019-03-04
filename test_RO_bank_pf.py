'''
Test the PF by RO System
'''
import torch
import numpy as np
import argparse
import logging
import matplotlib as mpl
import matplotlib.pyplot as plt
from bank_particle_filter import bpf
from data_manager import data_manager
from RO import RO
from utilities import obtain_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--index', type=int, help='choose the index in the data set')
    parser.add_argument('-t', '--test', type=str, help='test_set')
    args = parser.parse_args()
    mpl.rc('font',family='Times New Roman')
    # read parameters from environment
    index = 0 if args.index is None else args.index
    si = 0.01
    process_snr = 45
    obs_snr = 20
    test = 'test' if args.test is None else args.test

    data_cfg = '{}/RO.cfg'.format(test)
    data_mana = data_manager(data_cfg, si)
    msg = data_mana.get_info(index, prt=False)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)
    modes = data_mana.select_modes(index)

    state_sigma = np.sqrt(obtain_var(state, process_snr))
    state_sigma[-1] = 1
    obs_sigma = np.sqrt(obtain_var(output, obs_snr))

    logging.basicConfig(filename='log\\log.txt', level=logging.INFO)

    ro = RO(si)
    tracker = bpf(ro, state_sigma, obs_sigma)
    tracker.set_mode_num(3)
    # tracker.set_blend(True)
    tracker.log_msg(msg)
    tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=50)
    tracker.plot_state()
    tracker.plot_mode()
