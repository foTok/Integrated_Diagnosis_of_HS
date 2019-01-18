'''
Simulate RO system.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import matplotlib.pyplot as plt

class RO:
    # parameters
    I_fp    = 0.1 # N*s^2/m^5
    I_rp    = 2.0 # N*s^2/m^5
    R_fp    = 0.1 # N/m^5
    R_rp    = 0.1 # N/m^5
    C_k     = 565.0 # m^5/N
    R_forward   = 70.0 # N/m^5
    C_tr    = 1.5 # m^5/N
    R_return_l  = 15.0 # N/m^5
    R_return_s  = 8.0 # N/m^5
    R_return_AES    = 5.0 # N/m^5
    C_memb  = 0.6 # m^5/N
    C_brine = 8.0 # m^5/N
    p_fp    = 1.0 # N/m^2
    p_rp    = 160.0 # N/m^2
    # modes
    modes = ['normal', 'pressure', 'reverse', 's_normal', 's_pressure', 's_reverse']
    # states
    states = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # outputs
    outputs = ['q_fp', 'p_tr', 'q_rp', 'e_Cbrine', 'e_Ck']
    # vars
    variables = ['mode', 'q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    def __init__(self, step_len):
        self.step_len   = step_len
        # trajectory
        self.modes  = []
        self.states = []

    def fault_parameters(self, t, fault_type=None, fault_time=None, fault_magnitude=None):
        if (fault_time is None) or (t <= fault_time):
            return None, [0, 0, 0]
        if fault_type==3 or fault_type==4 or fault_type==5:
            return fault_type, [0, 0, 0]
        elif fault_type=='f_f':
            return None, [fault_magnitude, 0, 0]
        elif fault_type=='f_r':
            return None, [0, fault_magnitude, 0]
        elif fault_type=='f_m':
            return None, [0, 0, fault_magnitude]
        else:
            raise RuntimeError('Unknown Fault.')

    def run(self, init_state, t, fault_type=None, fault_time=None, fault_magnitude=None):
        i = 1
        mode_i = None
        state_i = init_state
        while i*self.step_len <= t:
            i += 1
            mode_i, state_i = self.mode_step(mode_i, state_i)
            self.modes.append(mode_i)
            mode_fault, para_fault = self.fault_parameters(i*self.step_len, fault_type, fault_time, fault_magnitude)
            mode_i = mode_i if mode_fault is None else mode_fault
            state_i, output_i = self.state_step(mode_i, state_i, para_fault)
            self.states.append(state_i)
            self.outputs.append(output_i)

    def mode_step(self, mode_i, state_i):
        h1 = 28.6770
        h2 = 17.2930
        h3 = 0.0670
        mode_ip1 = mode_i
        p = state_i[3]
        if mode_i is None:
            mode_ip1 = 0
        elif mode_i == 0:
            if p > h1:
                mode_ip1 = 1
        elif mode_i == 1:
            if p < h2:
                mode_ip1 = 2
        elif mode_i == 2:
            if p < h3:
                mode_ip1 = 0
                state_i = state_i[:]
                state_i[4] = 0
                state_i[5] = 0
        else:
            pass # keep the mode
        return mode_ip1, state_i

    def state_step(self, mode_ip1, state_i, fault_parameters):
        if (mode_ip1 % 3) == 0:
            _sigma1, _sigma2 = 1, 0
        elif (mode_ip1 % 3) == 1:
            _sigma1, _sigma2 = 0, 1
        elif (mode_ip1 % 3) == 2:
            _sigma1, _sigma2 = 0, 0
        else:
            pass # never
        # extract former state
        _q_fp, _p_tr, _q_rp, _p_memb, _e_Cbrine, _e_Ck = state_i
        # extract fault parameters
        f_f, f_r, f_m = fault_parameters
        # step forward
        step_len = self.step_len
        # e_RO20 in Chapter_13
        R_memb  = 0.202*(4.137e11*((_e_Ck - 12000)/165 + 29))
        # e_RO1
        q_fp    = _q_fp + step_len* \
                (- RO.R_fp*_q_fp \
                 - _p_tr \
                 + RO.p_fp*(1 - f_f)) \
                 /RO.I_fp
        # e_RO2
        p_tr    = _p_tr + step_len* \
                (_q_fp \
                 + _sigma1*(_p_memb - _p_tr)/RO.R_return_l \
                 - _q_rp \
                 + _sigma2*(_p_memb - _p_tr)/RO.R_return_s) \
                 /RO.C_tr
        # e_RO3
        d_q_rp  = step_len* \
                (- RO.R_rp*_q_rp \
                 - RO.R_forward*_q_rp \
                 - _p_memb \
                 + RO.p_rp*(1 - f_r))/RO.I_rp
        q_rp    = (_sigma1 + _sigma2)*(_q_rp + d_q_rp) \
                 + (1 - _sigma1 - _sigma2)*(_p_tr - _p_memb)/RO.R_forward
        # e_RO4
        p_memb  = _p_memb + step_len* \
                (_q_rp \
                 - _p_memb/(R_memb*(1 + f_m)) \
                 - _sigma1*(_p_memb - _p_tr)/RO.R_return_l \
                 - _sigma2*(_p_memb - _p_tr)/RO.R_return_s \
                 - (1 - _sigma1 - _sigma2)*_p_memb/RO.R_return_AES) \
                 / RO.C_memb
        # e_RO5
        e_Cbrine    = _e_Cbrine + step_len*\
                (_sigma1*(_p_memb - _p_tr)/RO.R_return_l \
                 + _sigma2*(_p_memb - _p_tr)/RO.R_return_s \
                 + (1 - _sigma1 - _sigma2)*_p_memb/RO.R_return_AES) \
                 / (1.667e-8*RO.C_brine)
        # e_RO6
        e_Ck    = _e_Ck + step_len* \
                _q_rp*(6*RO.C_brine + 0.1)/ (1.667e-8 * RO.C_k)
        
        # states and outputs
        states_ip1 = [q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck]
        output_ip1 = [q_fp, p_tr, q_rp, e_Cbrine, e_Ck]
        return states_ip1, output_ip1

    def np_modes(self):
        return np.array(self.modes)

    def np_states(self):
        return np.array(self.states)

    def np_outputs(self):
        return np.array(self.outputs)

    def np_data(self):
        modes = np.array(self.modes)
        states = np.array(self.states)
        modes = modes.reshape(len(modes), 1)
        data = np.concatenate((modes, states), 1)
        return data

    def _show(self, name):
        assert isinstance(name, str)
        if name in RO.states:
            i   = RO.states.index(name)
            y   = np.array(self.states)[:, i]
        elif name=='mode':
            y   = np.array(self.modes)
        else:
            raise RuntimeError('Unknown name.')
        x = (np.array(range(len(self.states)))+1)*self.step_len
        plt.figure()
        plt.plot(x, y)
        plt.xlabel('Time(s)')
        plt.ylabel(name)
        plt.show()

    def show(self, name=None):
        if name is not None:
            self._show(name)
        else:
            for name in RO.states:
                self._show(name)

    def reset(self):
        # trajectory
        self.modes  = []
        self.states = []
        self.outputs = []
