'''
Test the PF by RO System
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt
import time
from particle_fiter import chi2_confidence
from particle_fiter import exp_confidence
from particle_fiter import hs_system_wrapper
from particle_fiter import hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var

if __name__ == '__main__':
    si = 0.01
    process_snr = 40
    obs_snr = 20
    index = 107
    limit = (2, 3)
    norm_o=np.array([1,1,1,10e9,10e8])
    norm_s=np.array([1,1,1,30,10e9,10e8])
    identifier = os.path.join(parentdir, 'ANN\\RO\\train\\ro1.cnn')
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\train\\RO.cfg')
    data_mana = data_manager(data_cfg, si)
    data_mana.get_info(index)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)

    pv = obtain_var(state, process_snr)
    ov = obtain_var(output, obs_snr)
    states_sigma = np.sqrt([0.008, 0.002, 0.015, 0.002, 0.000, 0.003])
    paras_sigma = np.sqrt([0.002, 0.000, 0.006])

    ro = RO(si)
    hsw = hs_system_wrapper(ro, pv, ov*1.2)
    tracker = hpf(hsw)
    tracker.load_identifier(identifier)
    tracker.set_norm(norm_o, norm_s)
    tracker.set_sigma(states_sigma, paras_sigma)
    tracker.track(modes=0, state_mean=np.zeros(6), state_var=np.zeros(6), observations=output_with_noise, limit=limit, Nmin=150, Nmax=150)
    tracker.plot_states()
    tracker.plot_modes()
    tracker.plot_res()
    tracker.plot_Z()
    tracker.plot_paras()
