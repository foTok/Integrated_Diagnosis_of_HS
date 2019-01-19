'''
Test the PF by RO System
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt
from particle_fiter import hs_system_wrapper
from particle_fiter import chi2_hpf
from Systems.data_manager import data_manager
from Systems.RO_System.RO import RO

si = 0.01
data_cfg = parentdir + '\\Systems\\RO_System\\data\\debug\\0.cfg'
data_mana = data_manager(data_cfg)
data_mana.set_sample_int(si)
state = data_mana.select_states(0)
output = data_mana.select_outputs(0)
output_with_noise = data_mana.select_outputs(0, 0.01)

pv = np.mean(state, 0) *0.0001
ov = np.mean(output_with_noise, 0)

ro = RO(si)
hsw = hs_system_wrapper(ro, pv, ov)
tracker = chi2_hpf(hsw)
tracker.track(modes=0, state_mean=[0,0,0,0,0,0], state_var=[0,0,0,0,0,0], N=10, observations=output_with_noise)
tracker.find_best_trajectory()
tracker.plot(3)

print('Done')
