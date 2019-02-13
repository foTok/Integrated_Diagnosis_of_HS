'''
This script is used to generate data for training.
'''
import os
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootdir)
import numpy as np
import progressbar
import argparse
from RO import RO
from Systems.data_manager import cfg
from Systems.data_manager import term

def simulate(file_name, model, init_state=[0,0,0,0,0,0], t=300, sample_int=0.01, fault_type=None, fault_time=None, fault_magnitude=None):
    model.run(init_state, t, fault_type, fault_time, fault_magnitude)
    data = model.np_data()
    np.save(file_name, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--key', type=str, help='set the key word.')
    args = parser.parse_args()
    # read parameters from environment
    key = args.key
    this_path = os.path.dirname(os.path.abspath(__file__))
    fault_time_list = {'s_normal': np.concatenate((np.arange(128.23, 128.23+5.5, 0.5), \
                                   np.arange(128.23+5, 161.04-4.5, 2), \
                                   np.arange(161.04-5, 161.04+5.5, 0.5), \
                                   np.arange(161.04+5, 190.78-4.5, 2))),\
                       's_pressure':np.concatenate((np.arange(95.03, 95.03+5.5, 0.5), \
                                   np.arange(95.03+5, 128.23-4.5, 2), \
                                   np.arange(161.04, 161.04+5.5, 0.5), \
                                   np.arange(161.04+5, 190.78-4.5, 2), \
                                   np.arange(190.78-5, 190.78, 0.5))),\
                       's_reverse':np.concatenate((np.arange(95.03, 95.03+5.5, 0.5), \
                                   np.arange(95.03+5, 128.23-4.5, 2), \
                                   np.arange(128.23-5, 128.23+5.5, 0.5), \
                                   np.arange(128.23+5, 161.04-5, 2))),\
                       'f_f':np.concatenate((np.arange(95.03, 95.03+5.5, 0.5), \
                                   np.arange(95.03+5, 128.23-4.5, 2), \
                                   np.arange(128.23-5, 128.23+5.5, 0.5), \
                                   np.arange(128.23+5, 161.04-4.5, 2), \
                                   np.arange(161.04-5, 161.04+5.5, 0.5),
                                   np.arange(161.04+5, 190.78-4.5, 2),
                                   np.arange(190.78-5, 190.78, 0.5))),\
                       'f_m':np.concatenate((np.arange(95.03, 95.03+5.5, 0.5), \
                                   np.arange(95.03+5, 128.23-4.5, 2), \
                                   np.arange(128.23-5, 128.23+5.5, 0.5), \
                                   np.arange(128.23+5, 161.04-4.5, 2), \
                                   np.arange(161.04-5, 161.04+5.5, 0.5),
                                   np.arange(161.04+5, 190.78-4.5, 2),
                                   np.arange(190.78-5, 190.78, 0.5))), \
                       'f_r':np.concatenate((np.arange(95.03, 95.03+5.5, 0.5), \
                                   np.arange(95.03+5, 128.23-4.5, 2), \
                                   np.arange(128.23-5, 128.23+5.5, 0.5), \
                                   np.arange(128.23+5, 161.04-4.5, 2)))}
    fault_type_list = ['s_normal', 's_pressure', 's_reverse', 'f_f', 'f_m', 'f_r']
    fault_magnitude_list = np.arange(0.05, 0.505, 0.05)
    file_num = 1
    for f in fault_type_list:
        if f.startswith('s'):
            file_num += len(fault_time_list[f])
        else:
            file_num += len(fault_time_list[f])*len(fault_magnitude_list)
    i = 0 # file index
    sample_int = 0.01
    progressbar.streams.wrap_stderr()
    with progressbar.ProgressBar(max_value=100) as bar:
        # normal
        file_name = os.path.join(this_path, 'data\\{}\\{}'.format(key, i))
        cfg_name = os.path.join(this_path, 'data\\{}\\RO.cfg'.format(key))
        i += 1
        bar.update(min(float('%.2f'%(i*100/file_num)), 100))
        path = os.path.dirname(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        the_cfg = cfg(RO.modes, RO.states, RO.outputs, RO.variables, RO.f_parameters, RO.labels, sample_int)
        the_cfg.add_term(term(file_name))
        ro = RO(sample_int)
        simulate(file_name, ro, sample_int=sample_int)
        # fault
        for fault_type in fault_type_list:
            for fault_time in fault_time_list[fault_type]:
                _fault_magnitude_list = [None] if fault_type.startswith('s') else fault_magnitude_list
                for fault_magnitude in _fault_magnitude_list:
                    file_name = os.path.join(this_path, 'data\\{}\\{}'.format(key, i))
                    i += 1
                    bar.update(min(float('%.2f'%(i*100/file_num)), 100))
                    ro = RO(sample_int)
                    the_term = term(file_name, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
                    the_cfg.add_term(the_term)
                    simulate(file_name, ro, sample_int=sample_int, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
        bar.update(100)
        progressbar.streams.flush()
    the_cfg.save(cfg_name)
