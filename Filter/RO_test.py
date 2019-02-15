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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--ann', type=str, help='choose ann type')
    parser.add_argument('-i', '--index', type=int, help='choose the index in the data set')
    parser.add_argument('-fd', '--fd', type=float, help='fault detection close window')
    parser.add_argument('-pf', '--pf', type=float, help='particle filter close window')
    parser.add_argument('-fp', '--fp', type=float, help='fault parameter estimation window')
    args = parser.parse_args()
    # read parameters from environment
    ann = args.ann
    index = args.index
    fd, pf, fp = (10 if args.fd is None else args.fd), (0 if args.pf is None else args.pf), (8 if args.fp is None else args.fp)
    si = 0.01
    process_snr = 45
    obs_snr = 20
    limit = (2, 3)
    proportion = 1.0
    state_scale =np.array([1,1,1,30,10e9,10e8])
    obs_scale =np.array([1,1,1,10e9,10e8])
    identifier = os.path.join(parentdir, 'ANN\\RO\\train\\ro.{}'.format(ann))
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\test\\RO.cfg')
    data_mana = data_manager(data_cfg, si)
    data_mana.get_info(index)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)

    state_sigma = np.sqrt(obtain_var(state, process_snr))
    obs_sigma = np.sqrt(obtain_var(output, obs_snr))

    ro = RO(si)
    hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
    tracker = hpf(hsw)
    tracker.load_identifier(identifier)
    tracker.set_scale(state_scale, obs_scale)
    tracker.track(modes=0, state_mean=np.zeros(6), state_var=np.zeros(6), \
                  observations=output_with_noise, limit=limit, \
                  fd=fd, pf=pf, fp=fp, proportion=proportion, \
                  Nmin=150, Nmax=200)
    tracker.plot_states()
    tracker.plot_modes()
    tracker.plot_res()
    tracker.plot_Z()
    tracker.plot_paras()
