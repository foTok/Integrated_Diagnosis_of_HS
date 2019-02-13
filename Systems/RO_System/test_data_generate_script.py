'''
This script is used to generate data for training.
'''
import os
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootdir)
import numpy as np
import progressbar
from RO import RO
from numpy.random import uniform
from Systems.data_manager import cfg
from Systems.data_manager import term

def simulate(file_name, model, init_state=[0,0,0,0,0,0], t=300, sample_int=0.01, fault_type=None, fault_time=None, fault_magnitude=None):
    model.run(init_state, t, fault_type, fault_time, fault_magnitude)
    data = model.np_data()
    np.save(file_name, data)

if __name__ == "__main__":
    this_path = os.path.dirname(os.path.abspath(__file__))
    fault_time_list = {'s_normal': [ round(t, 2) for t in uniform(128, 190, 2)], \
                       's_pressure':[ round(t, 2) for t in uniform(65, 128, 2)], \
                       's_reverse':[ round(t, 2) for t in uniform(95, 162, 2)], \
                       'f_f':[ round(t, 2) for t in uniform(90, 220, 3)], \
                       'f_r':[ round(t, 2) for t in uniform(90, 220, 3)]}
    fault_type_list = ['s_normal', 's_pressure', 's_reverse', 'f_f', 'f_r']
    fault_magnitude_list = [ round(p, 2) for p in uniform(0.1, 0.5, 3)]
    file_num = 1
    for f in fault_type_list:
        if f.startswith('s'):
            file_num += len(fault_time_list[f])
        else:
            file_num += len(fault_time_list[f])*len(fault_magnitude_list)
    i = 0 # file index
    sample_int = 0.01
    progressbar.streams.wrap_stderr()
    readme = []
    with progressbar.ProgressBar(max_value=100) as bar:
        # normal
        file_name = os.path.join(this_path, 'data\\test\\{}'.format(i))
        cfg_name = os.path.join(this_path, 'data\\test\\RO.cfg')
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
                    file_name = os.path.join(this_path, 'data\\test\\{}'.format(i))
                    msg = 'file: {}, fault_type: {}, fault_time: {}, fault_managnitude: {}.\n'.format(i, fault_type, fault_time, fault_magnitude)
                    readme.append(msg)
                    i += 1
                    bar.update(min(float('%.2f'%(i*100/file_num)), 100))
                    ro = RO(sample_int)
                    the_term = term(file_name, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
                    the_cfg.add_term(the_term)
                    simulate(file_name, ro, sample_int=sample_int, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
        bar.update(100)
        progressbar.streams.flush()
    the_cfg.save(cfg_name)
    with open(os.path.join(this_path, 'data\\test\\readme.txt'), 'w') as f:
        f.writelines(readme)
