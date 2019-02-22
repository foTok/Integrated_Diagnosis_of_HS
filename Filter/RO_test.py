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
    parser.add_argument('-fd', '--fd', type=float, help='fault detection close window')
    parser.add_argument('-fp', '--fp', type=float, help='fault parameter estimation window')
    args = parser.parse_args()
    # read parameters from environment
    conf = exp_confidence if args.conf=='exp' else chi2_confidence
    index = 0 if args.index is None else args.index
    fd, fp = (3 if args.fd is None else args.fd), (8 if args.fp is None else args.fp)
    si = 0.01
    process_snr = 45
    obs_snr = 20
    limit = (3, 2)
    proportion = 1.0
    state_scale =np.array([1,1,1,30,10e9,10e8])
    obs_scale =np.array([1,1,1,10e9,10e8])
    test = 'test' if args.test is None else args.test
    identifier = os.path.join(parentdir, 'ANN\\RO\\train\\ro.cnn')
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\{}\\RO.cfg'.format(test))
    data_mana = data_manager(data_cfg, si)
    msg = data_mana.get_info(index, prt=False)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)
    modes = data_mana.select_modes(index)

    state_sigma = np.sqrt(obtain_var(state, process_snr))
    obs_sigma = np.sqrt(obtain_var(output, obs_snr))

    ro = RO(si)
    hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
    logging.basicConfig(filename='log\\log.txt', level=logging.INFO)
    tracker = hpf(hsw, conf=conf)
    tracker.load_identifier(identifier)
    tracker.set_scale(state_scale, obs_scale)
    tracker.log_msg(msg)
    tracker.track(modes=0, state_mean=np.zeros(6), state_var=np.zeros(6), \
                  observations=output_with_noise, limit=limit, \
                  fd=fd, fp=fp, proportion=proportion, \
                  Nmin=150, Nmax=200)
    tracker.plot_states(file_name='log/states')
    tracker.plot_modes(file_name='log/modes')
    tracker.plot_res(file_name='log/res')
    tracker.plot_Z(file_name='log/Z')
    tracker.plot_paras(file_name='log/paras')
