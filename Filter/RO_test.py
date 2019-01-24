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
from particle_fiter import hs_system_wrapper
from particle_fiter import chi2_hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO
from utilities.utilities import obtain_var

if __name__ == '__main__':
    si = 0.001
    process_snr = 70
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
    ro.set_state_disturb(pv)
    hsw = hs_system_wrapper(ro, pv*1.2, ov*1.2)
    tracker = chi2_hpf(hsw)
    start = time.clock()
    tracker.track(modes=0, state_mean=[0,0,0,0,0,0], state_var=[0,0,0,0,0,0], N=30, observations=output_with_noise, parallel=True)
    elapsed = (time.clock() - start)
    print("Time used:", elapsed)
    for i in range(5):
        tracker.plot(i)
    tracker.plot_mode()

    print('Done')
