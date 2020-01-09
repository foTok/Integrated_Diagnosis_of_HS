'''
Test the PF by RO System
'''
import torch
import numpy as np
import argparse
import logging
import matplotlib as mpl
from integrated_particle_filter import ipf
from data_manager import data_manager
from RO import RO
from utilities import obtain_var
from cycler import cycler

mpl.rcParams['axes.prop_cycle'] = cycler(color='kbgrcmy')

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
    obs_scale =np.array([1,1,1,10,10e9])
    test = 'test' if args.test is None else args.test
    mode_detector = 'model/mode_detector'
    pf_isolator = 'model/pf_isolator'
    f_f_identifier = 'model/f_f_identifier'
    f_r_identifier = 'model/f_r_identifier'
    f_m_identifier = 'model/f_m_identifier'

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

    with torch.no_grad():
        ro = RO(si)
        tracker = ipf(ro, state_sigma, obs_sigma)
        tracker.load_mode_detector(mode_detector)
        tracker.load_pf_isolator(pf_isolator)
        tracker.load_identifier([f_f_identifier, f_r_identifier, f_m_identifier])
        tracker.set_scale(obs_scale)
        tracker.log_msg(msg)
        tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=50, Nmax=100)
        tracker.plot_state()
        tracker.plot_mode()
        tracker.plot_res()
        tracker.plot_para()
        tracker.plot_para_fault()
