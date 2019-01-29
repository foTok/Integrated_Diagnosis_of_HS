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
    process_snr = 35
    obs_snr = 20
    data_cfg = parentdir + '\\Systems\\RO_System\\data\\debug\\0.cfg'
    data_mana = data_manager(data_cfg)
    data_mana.set_sample_int(si)
    state = data_mana.select_states(0)
    state_with_noise = data_mana.select_states(0, process_snr)
    output = data_mana.select_outputs(0)
    output_with_noise = data_mana.select_outputs(0, obs_snr)

    pv = obtain_var(state, process_snr)
    ov = obtain_var(output, obs_snr)

    ro = RO(si)
    # ro.set_state_disturb(pv)
    hsw = hs_system_wrapper(ro, pv, ov*1.5)
    tracker = hpf(hsw)
    tracker.track(modes=0, state_mean=[0,0,0,0,0,0], state_var=[0,0,0,0,0,0], observations=output_with_noise, Nmin=100, Nmax=150)
    tracker.plot_states()
    tracker.plot_modes()
    tracker.plot_res()
