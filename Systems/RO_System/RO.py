'''
Simulate RO system.
'''
import os
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootdir)
import numpy as np
import matplotlib.pyplot as plt
from utilities.utilities import add_noise

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
    modes = {'mode':['normal', 'pressure', 'reverse', 's_normal', 's_pressure', 's_reverse']}
    # states
    states = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # outputs
    outputs = ['q_fp', 'p_tr', 'q_rp', 'e_Cbrine', 'e_Ck']
    # vars
    variables = ['mode', 'q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # fault parameters
    f_parameters = ['f_f', 'f_r', 'f_m']
    # labels
    labels = ['normal', 's_normal', 's_pressure', 's_reverse', 'f_f', 'f_r', 'f_m']
    def __init__(self, step_len): # important interface
        self.step_len   = step_len
        # trajectory
        self.modes  = []
        self.states = []
        self.outputs = []
        self.state_disturb = None

    def set_state_disturb(self, disturb): # important interface
        self.state_disturb = disturb

    def fault_parameters(self, t, mode, fault_type=None, fault_time=None, fault_magnitude=None): # important interface
        if (fault_time is None) or (t <= fault_time): # no fault
            return mode, [0, 0, 0]
        if fault_type=='s_normal':
            return 3, [0, 0, 0]
        elif fault_type=='s_pressure':
            return 4, [0, 0, 0]
        elif fault_type=='s_reverse':
            return 5, [0, 0, 0]
        elif fault_type=='f_f':
            return mode, [fault_magnitude, 0, 0]
        elif fault_type=='f_r':
            return mode, [0, fault_magnitude, 0]
        elif fault_type=='f_m':
            return mode, [0, 0, fault_magnitude]
        else:
            raise RuntimeError('Unknown Fault.')

    def run(self, init_state=[0, 0, 0, 0, 0, 0], t=0, fault_type=None, fault_time=None, fault_magnitude=None): # importance interface
        i = 1
        mode_i = None
        state_i = init_state
        while i*self.step_len <= t:
            i += 1
            # if insert fault
            mode_i, para_fault = self.fault_parameters(i*self.step_len, mode_i, fault_type, fault_time, fault_magnitude)
            mode_i, state_i = self.mode_step(mode_i, state_i) # mode +1
            state_i = self.state_step(mode_i, state_i, para_fault) # state +1
            output_i = self.output(mode_i, state_i) # output
            self.modes.append(mode_i)
            self.states.append(state_i)
            self.outputs.append(output_i)

    def close2switch(self, mode, state): # important interface
        h1 = 28.6770
        h2 = 17.2930
        h3 = 0.0670
        p = state[3]
        if mode==0:
            if abs(p-h1)<1 or abs(p-h3)<4:
                return True
        elif mode==1:
            if abs(p-h2)<0.5 or abs(p-h1)<0.5:
                return True
        elif mode==2:
            if abs(p-h2)<1 or abs(p-h3)<1:
                return True
        return False

    def mode_step(self, mode_i, state_i): # important interface
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

    def state_step(self, mode_ip1, state_i, fault_parameters): # important interface
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
        if self.state_disturb is not None:
            states_ip1 = add_noise(states_ip1, self.state_disturb)
        return states_ip1

    def output(self, mode, states): # important interface
        q_fp, p_tr, q_rp, _, e_Cbrine, e_Ck = states
        return [q_fp, p_tr, q_rp, e_Cbrine, e_Ck]

    def np_modes(self):
        return np.array(self.modes)

    def np_states(self):
        return np.array(self.states)

    def np_outputs(self):
        return np.array(self.outputs)

    def np_data(self): # important interface
        modes = np.array(self.modes)
        states = np.array(self.states)
        modes = modes.reshape(len(modes), 1)
        data = np.concatenate((modes, states), 1)
        return data

    def plot_states(self, states=None, file_name=None): # important interface
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('System States')  # Add a title so we know which it is
        data = self.np_states() if states is None else states
        x = np.arange(len(data))*self.step_len
        # 0
        ax_lst[0, 0].plot(x, data[:, 0])
        ax_lst[0, 0].set_ylabel(RO.states[0])
        plt.setp(ax_lst[0, 0].get_xticklabels(), visible=False)
        # 1
        ax_lst[1, 0].plot(x, data[:, 1])
        ax_lst[1, 0].set_ylabel(RO.states[1])
        plt.setp(ax_lst[1, 0].get_xticklabels(), visible=False)
        # 2
        ax_lst[2, 0].plot(x, data[:, 2])
        ax_lst[2, 0].set_xlabel('Time/s')
        ax_lst[2, 0].set_ylabel(RO.states[2])
        # 3
        ax_lst[0, 1].plot(x, data[:, 3])
        ax_lst[0, 1].set_ylabel(RO.states[3])
        plt.setp(ax_lst[0, 1].get_xticklabels(), visible=False)
        # 4
        ax_lst[1, 1].plot(x, data[:, 4])
        ax_lst[1, 1].set_ylabel(RO.states[4])
        plt.setp(ax_lst[1, 1].get_xticklabels(), visible=False)
        # 5
        ax_lst[2, 1].plot(x, data[:, 5])
        ax_lst[2, 1].set_xlabel('Time/s')
        ax_lst[2, 1].set_ylabel(RO.states[5])
        plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name if file_name.endswith('.svg') else (file_name+'.svg'), format='svg')
        plt.close()

    def plot_modes(self, modes=None, file_name=None): # important interface
        data = self.np_modes() if modes is None else modes
        mode_labels = ['normal', 'pressure', 'reverse', 's_normal', 's_pressure', 's_reverse']
        max_mode = int(max(data))
        y_ticks_pos = range(max_mode+1)
        y_ticks_label = mode_labels[:max_mode+1]
        x = np.arange(len(data))*self.step_len
        plt.plot(x, data)
        plt.xlabel('Time/s')
        plt.ylabel('Mode')
        plt.yticks(y_ticks_pos, y_ticks_label)
        plt.title('System Modes')
        plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name if file_name.endswith('.svg') else (file_name+'.svg'), format='svg')
        plt.close()

    def plot_res(self, data, file_name=None): # important interface
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 3x2 grid of Axes
        fig.suptitle('Residuals')  # Add a title so we know which it is
        x = np.arange(len(data))*self.step_len
        # 0
        ax_lst[0, 0].plot(x, data[:, 0])
        ax_lst[0, 0].set_ylabel('r1')
        plt.setp(ax_lst[0, 0].get_xticklabels(), visible=False)
        # 1
        ax_lst[1, 0].plot(x, data[:, 1])
        ax_lst[1, 0].set_ylabel('r2')
        plt.setp(ax_lst[1, 0].get_xticklabels(), visible=False)
        # 2
        ax_lst[2, 0].plot(x, data[:, 2])
        ax_lst[2, 0].set_xlabel('Time/s')
        ax_lst[2, 0].set_ylabel('r3')
        # 3
        ax_lst[0, 1].plot(x, data[:, 3])
        ax_lst[0, 1].set_ylabel('r4')
        plt.setp(ax_lst[0, 1].get_xticklabels(), visible=False)
        # 4
        ax_lst[1, 1].plot(x, data[:, 4])
        ax_lst[1, 1].set_ylabel('r5')
        ax_lst[1, 1].set_xlabel('Time/s')
        # 5
        fig.delaxes(ax_lst[2, 1])
        plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name if file_name.endswith('.svg') else (file_name+'.svg'), format='svg')
        plt.close()

    def plot_Z(self, data, file_name): # important interface
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 3x2 grid of Axes
        fig.suptitle('Z values')  # Add a title so we know which it is
        x = np.arange(len(data))*self.step_len
        # 0
        ax_lst[0, 0].plot(x, data[:, 0])
        ax_lst[0, 0].set_ylabel('z1')
        plt.setp(ax_lst[0, 0].get_xticklabels(), visible=False)
        ax_lst[0, 0].set_yticks([0, 1])
        ax_lst[0, 0].set_yticklabels([0, 1])
        # 1
        ax_lst[1, 0].plot(x, data[:, 1])
        ax_lst[1, 0].set_ylabel('z2')
        plt.setp(ax_lst[1, 0].get_xticklabels(), visible=False)
        ax_lst[1, 0].set_yticks([0, 1])
        ax_lst[1, 0].set_yticklabels([0, 1])
        # 2
        ax_lst[2, 0].plot(x, data[:, 2])
        ax_lst[2, 0].set_xlabel('Time/s')
        ax_lst[2, 0].set_ylabel('z3')
        ax_lst[2, 0].set_yticks([0, 1])
        ax_lst[2, 0].set_yticklabels([0, 1])
        # 3
        ax_lst[0, 1].plot(x, data[:, 3])
        ax_lst[0, 1].set_ylabel('z4')
        plt.setp(ax_lst[0, 1].get_xticklabels(), visible=False)
        ax_lst[0, 1].set_yticks([0, 1])
        ax_lst[0, 1].set_yticklabels([0, 1])
        # 4
        ax_lst[1, 1].plot(x, data[:, 4])
        ax_lst[1, 1].set_ylabel('z5')
        ax_lst[1, 1].set_xlabel('Time/s')
        ax_lst[1, 1].set_yticks([0, 1])
        ax_lst[1, 1].set_yticklabels([0, 1])
        # 5
        fig.delaxes(ax_lst[2, 1])
        plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name if file_name.endswith('.svg') else (file_name+'.svg'), format='svg')
        plt.close()

    def plot_paras(self, data, file_name=None):
        fig, ax_lst = plt.subplots(3, 1)  # A figure with a 3x2 grid of Axes
        fig.suptitle('Fault parameters')  # Add a title so we know which it is
        x = np.arange(len(data))*self.step_len
        # 0
        ax_lst[0].plot(x, data[:, 0])
        ax_lst[0].set_ylabel('f_f')
        plt.setp(ax_lst[0].get_xticklabels(), visible=False)
        # 1
        ax_lst[1].plot(x, data[:, 1])
        ax_lst[1].set_ylabel('f_r')
        plt.setp(ax_lst[1].get_xticklabels(), visible=False)
        # 2
        ax_lst[2].plot(x, data[:, 2])
        ax_lst[2].set_xlabel('Time/s')
        ax_lst[2].set_ylabel('f_m')
        plt.tight_layout(pad=2.0, w_pad=1.0, h_pad=1.0)
        if file_name is None:
            plt.show()
        else:
            plt.savefig(file_name if file_name.endswith('.svg') else (file_name+'.svg'), format='svg')
        plt.close()
