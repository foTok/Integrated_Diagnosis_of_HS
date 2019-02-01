import os
import numpy as np
import argparse
from RO_System.RO import RO
from C130FS.C130FS import C130FS
from data_manager import cfg
from data_manager import term

this_path = os.path.dirname(os.path.abspath(__file__))

def simulate(file_name, model, init_state=[0,0,0,0,0,0], t=300, sample_int=0.001, fault_type='s_reverse', fault_time=20, fault_magnitude=0):
    model.run(init_state, t, fault_type, fault_time, fault_magnitude)
    data = model.np_data()
    np.save(file_name, data)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--system", type=str, choices=['RO', 'C130FS'], help="choose the system")
    args = parser.parse_args()
    if args.system=='RO':
        # for debug
        sample_int = 0.001
        file_name = os.path.join(this_path, 'RO_System\\data\\debug\\10')
        path = os.path.dirname(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        ro = RO(sample_int)
        cfg = cfg(RO.modes, RO.states, RO.outputs, RO.variables, sample_int)
        cfg.add_term(term(file_name))
        cfg.save(file_name)
        simulate(file_name, ro, sample_int=sample_int)
    elif args.system=='C130FS':
                # for debug
        sample_int = 0.1
        file_name = os.path.join(this_path, 'C130FS\\data\\debug\\2')
        path = os.path.dirname(file_name)
        if not os.path.isdir(path):
            os.makedirs(path)
        c130fs = C130FS(sample_int)
        cfg = cfg(C130FS.modes, C130FS.states, C130FS.outputs, C130FS.variables, sample_int)
        cfg.add_term(term(file_name))
        cfg.save(file_name)
        simulate(file_name, c130fs, init_state=[1340, 1230, 1230, 1340, 900, 900], sample_int=sample_int)
