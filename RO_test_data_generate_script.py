'''
This script is used to generate data for training.
'''
import os
import numpy as np
import progressbar
import argparse
from RO import RO
from numpy.random import uniform
from data_manager import cfg
from data_manager import term

def simulate(file_name, model, init_state=[0,0,0,0,0,0], t=300, sample_int=0.01, fault_type=None, fault_time=None, fault_magnitude=None):
    model.run(0, init_state, t, fault_type, fault_time, fault_magnitude)
    data = model.np_data()
    np.save(file_name, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--fault_type", type=str, choices=['norm', 'dis', 'cont'], help="set fault type.")
    parser.add_argument("-o", "--output", type=str, help="output directory.")
    args = parser.parse_args()
    out_dir = args.output
    assert out_dir is not None
    this_path = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(this_path, out_dir)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    fault_time_list = {'s_normal': [round(uniform(140, 145), 2),  round(uniform(170, 175), 2)], \
                       's_pressure':[round(uniform(110, 115), 2),  round(uniform(170, 175), 2)], \
                       's_reverse':[round(uniform(110, 115), 2),  round(uniform(140, 145), 2)], \
                       'f_f':[round(uniform(110, 115), 2),  round(uniform(140, 145), 2),  round(uniform(170, 175), 2)], \
                       'f_r':[round(uniform(110, 115), 2),  round(uniform(140, 145), 2)], \
                       'f_m':[round(uniform(110, 115), 2),  round(uniform(140, 145), 2),  round(uniform(170, 175), 2)],}
    dis_fault = ['s_normal', 's_pressure', 's_reverse']
    cont_fault = ['f_f', 'f_r', 'f_m']
    fault_magnitude_list = [0.16, 0.26, 0.36]

    if args.fault_type=='norm':
        file_num = 1
    elif args.fault_type=='dis':
        file_num = 0
        for f in dis_fault:
            file_num += len(fault_time_list[f])
    elif args.fault_type=='cont':
        file_num = 0
        for f in cont_fault:
            file_num += len(fault_time_list[f])*len(fault_magnitude_list)
    else:
        raise RuntimeError('Unknow fault type.')
    sample_int = 0.01
    readme = []

    if args.fault_type=='norm':
        # normal
        file_name = os.path.join(out_dir, '0')
        cfg_name = os.path.join(out_dir, 'RO.cfg')
        the_cfg = cfg(RO.modes, RO.states, RO.outputs, RO.variables, RO.f_parameters, RO.labels, sample_int)
        the_cfg.add_term(term(file_name))
        ro = RO(sample_int)
        msg = 'file 0, normal.'
        readme.append(msg)
        simulate(file_name, ro, sample_int=sample_int)
    else:
        i = -1 # file index
        fault_type_list = dis_fault if args.fault_type=='dis' else cont_fault
        fault_magnitude_list = [None] if args.fault_type=='dis' else fault_magnitude_list
        cfg_name = os.path.join(out_dir, 'RO.cfg')
        the_cfg = cfg(RO.modes, RO.states, RO.outputs, RO.variables, RO.f_parameters, RO.labels, sample_int)
        with progressbar.ProgressBar(max_value=100) as bar:
            for fault_type in fault_type_list:
                for fault_time in fault_time_list[fault_type]:
                    for fault_magnitude in fault_magnitude_list:
                        i += 1
                        file_name = os.path.join(out_dir, str(i))
                        msg = 'file: {}, fault_type: {}, fault_time: {}, fault_managnitude: {}.\n'.format(i, fault_type, fault_time, fault_magnitude)
                        readme.append(msg)
                        bar.update(min(float('%.2f'%(i*100/file_num)), 100))

                        ro = RO(sample_int)
                        the_term = term(file_name, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
                        the_cfg.add_term(the_term)
                        simulate(file_name, ro, sample_int=sample_int, fault_type=fault_type, fault_time=fault_time, fault_magnitude=fault_magnitude)
            bar.update(100)
            progressbar.streams.flush()
    the_cfg.save(cfg_name)
    with open(os.path.join(out_dir, 'readme.txt'), 'w') as f:
        f.writelines(readme)
