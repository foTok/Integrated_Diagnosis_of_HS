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
from particle_fiter import hpf
from Systems.data_manager import data_manager
from Systems.C130FS.C130FS import C130FS
from utilities.utilities import obtain_var

if __name__ == '__main__':
    si = 1
    process_snr = 60
    obs_snr = 20
    data_cfg = parentdir + '\\Systems\\C130FS\\data\\debug\\0.cfg'
    data_mana = data_manager(data_cfg)
    data_mana.set_sample_int(si)
    state = data_mana.select_states(0)
    state_with_noise = data_mana.select_states(0, process_snr)
    output = data_mana.select_outputs(0)
    output_with_noise = data_mana.select_outputs(0, obs_snr)

    pv = obtain_var(state, process_snr)
    ov = obtain_var(output, obs_snr)

    c130fs = C130FS(si)
    # c130fs.set_state_disturb(pv)
    hsw = hs_system_wrapper(c130fs, pv*1.2, ov*1.2)
    tracker = hpf(hsw)
    tracker.track(modes=([1,1,1,1]+[0]*8), state_mean=[1340, 1230, 1230, 1340, 900, 900], state_var=[0,0,0,0,0,0], N=30, observations=output_with_noise)
    tracker.plot_states()
    tracker.plot_modes(200)
    tracker.plot_res()
