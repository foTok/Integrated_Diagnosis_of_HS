'''
Test the Data Manager.
'''
import numpy as np
import matplotlib.pyplot as plt
from data_manager import data_manager
from RO import RO
from utilities import obtain_var

if __name__ == '__main__':
    si = 0.01
    process_snr = 50
    obs_snr = 20
    data_cfg = parentdir + '\\Systems\\RO_System\\data\\debug\\RO.cfg'
    data_mana = data_manager(data_cfg, si)
    state = data_mana.select_states(0)
    state_with_noise = data_mana.select_states(0, process_snr)
    output = data_mana.select_outputs(0)
    output_with_noise = data_mana.select_outputs(0, obs_snr)
    
    hs0, x, m, y, p = data_mana.sample(size=20, window=5, limit=(2,2), normal_proportion=0.2)
    torch_m = data_mana.np2target(m)
