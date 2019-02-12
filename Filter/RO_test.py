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
    index = 106
    limit = (2, 3)
    proportion = 0.85
    state_scale =np.array([1,1,1,30,10e9,10e8])
    obs_scale =np.array([1,1,1,10e9,10e8])
    identifier = os.path.join(parentdir, 'ANN\\RO\\train\\ro0.cnn2')
    data_cfg = os.path.join(parentdir, 'Systems\\RO_System\\data\\train\\RO.cfg')
    data_mana = data_manager(data_cfg, si)
    data_mana.get_info(index)
    state = data_mana.select_states(index)
    state_with_noise = data_mana.select_states(index, process_snr)
    output = data_mana.select_outputs(index)
    output_with_noise = data_mana.select_outputs(index, obs_snr)

    state_sigma = np.sqrt(obtain_var(state, process_snr))
    obs_sigma = np.sqrt(obtain_var(output, obs_snr))
    paras_sigma = np.sqrt([0.002, 0.000, 0.006])

    ro = RO(si)
    hsw = hs_system_wrapper(ro, state_sigma, obs_sigma)
    tracker = hpf(hsw)
    tracker.load_identifier(identifier)
    tracker.set_scale(state_scale, obs_scale)
    tracker.set_paras_sigma(paras_sigma)
    tracker.track(modes=0, state_mean=np.zeros(6), state_var=np.zeros(6), \
                  observations=output_with_noise, limit=limit, proportion=proportion, \
                  Nmin=150, Nmax=150)
    tracker.plot_states()
    tracker.plot_modes()
    tracker.plot_res()
    tracker.plot_Z()
    tracker.plot_paras()
