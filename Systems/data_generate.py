import os
import numpy as np
from RO_System.RO import RO
from C130FS.C130FS import C130FS
from data_manager import cfg
from data_manager import term

def simulate(file_name, model, init_state=[0,0,0,0,0,0], t=300, sample_int=0.001, fault_type=None, fault_time=None, fault_magnitude=None):
    model.run(init_state, t, fault_type, fault_time, fault_magnitude)
    data = model.np_data()
    np.save(file_name, data)

if __name__ == "__main__":
    # for debug
    sample_int = 0.001
    file_name = 'RO_System\\data\\debug\\0'
    path = os.path.dirname(file_name)
    if not os.path.isdir(path):
        os.makedirs(path)
    ro = RO(sample_int)
    cfg = cfg(RO.modes, RO.states, RO.outputs, RO.variables, sample_int)
    cfg.add_term(term(file_name))
    simulate(file_name, ro, sample_int=sample_int)
