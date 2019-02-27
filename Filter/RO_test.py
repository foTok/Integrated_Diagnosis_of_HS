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
from particle_fiter import chi2_confidence
from particle_fiter import exp_confidence
from particle_fiter import hs_system_wrapper
from particle_fiter import hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--conf', type=str, choices=['exp', 'chi2'], help='confidence')
    parser.add_argument('-i', '--index', type=int, help='choose the index in the data set')
    parser.add_argument('-t', '--test', type=str, help='test_set')
    args = parser.parse_args()
    # read parameters from environment
    conf = exp_confidence if args.conf=='exp' else chi2_confidence
    index = 0 if args.index is None else args.index
    si = 0.01
    process_snr = 45
    obs_snr = 20
    obs_scale =np.array([1,1,1,10,10e9])
    test = 'test' if args.test is None else args.test
    mode_detector = os.path.join(parentdir, 'ANN/model/cnn_gru_mode_detector')
    pf_isolator = os.path.join(parentdir, 'ANN/model/cnn_gru_pf_isolator_res')
    f_f_identifier = os.path.join(parentdir, 'ANN/model/cnn_gru_f_f_identifier_res')
    f_r_identifier = os.path.join(parentdir, 'ANN/model/cnn_gru_f_r_identifier_res')

    data_cfg = os.path.join(parentdir, 'Systems/RO_System/{}/RO.cfg'.format(test))
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

    ro = RO(si)
    hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
    logging.basicConfig(filename='log\\log.txt', level=logging.INFO)
    tracker = hpf(hsw, conf=conf)
    tracker.load_mode_detector(mode_detector)
    tracker.load_pf_isolator(pf_isolator)
    tracker.load_identifier([f_f_identifier, f_r_identifier])
    tracker.set_scale(obs_scale)
    tracker.log_msg(msg)
    tracker.track(mode=0, state_mu=np.zeros(6), state_sigma=np.zeros(6), obs=output_with_noise, N=50)
    tracker.plot_state()
    tracker.plot_mode()
    tracker.plot_res()
    tracker.plot_para()
    tracker.plot_para_fault()
