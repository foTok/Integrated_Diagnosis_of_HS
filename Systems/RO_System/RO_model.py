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
    # states
    states  = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # obs
    obs = {'y1':'q_fp', 'y2':'p_memb', 'y3':'q_fp', 'y4':'e_Cbrine', 'y5':'e_Ck'}

    def __init__(self, step_len=1.0):
        self.step_len   = step_len
        # fault parameters
        self.f_f    = 0
        self.f_r    = 0
        self.f_m    = 0
        # discrete modes
        self.sigma1 = None
        self.sigma2 = None
        # continous states
        self.inital_states = {'q_fp':0, 'q_rp':0, 'p_tr':0, 'p_memb':0, 'e_Cbrine':0, 'e_Ck': 0}
        self.q_fp   = 0
        self.p_tr   = 0
        self.q_rp   = 0
        self.p_memb = 0
        self.e_Cbrine   = 0
        self.e_Ck   = 0
        # trajectory
        self.modes  = []
        self.states = []
        self.para_faults    = []
        self.x  = []
        # how to transit
        self.t = 'state'
        # tmp
        self.tmp = 0

    def init(self, states):
        '''
        states: a dict to store the initial states
            flow: f or q
        '''
        for s in states:
            if s=='q_fp' or s=='f_fp':
                self.q_fp = states[s]
            elif s=='q_rp' or s=='f_rp':
                self.q_rp = states[s]
            elif s=='p_tr':
                self.p_tr = states[s]
            elif s=='p_memb':
                self.p_memb = states[s]
            elif s=='e_Cbrine':
                self.e_Cbrine = states[s]
            elif s== 'e_Ck':
                self.e_Ck = states[s]
            else:
                raise RuntimeError('Unknown States')
        self.inital_states = states
    
    def set_trans_mode(self, tm):
        self.t = tm

    def run(self, t):
        i = 1
        while i*self.step_len < t:
            i += 1
            self.mode_step(i)
            self.state_step(self.step_len)

    def mode_step(self, i):
        if self.sigma1 is None or self.sigma2 is None:
            self.sigma1, self.sigma2 = 1, 0
        if self.t == 'state':
            h1 = 28.6770
            h2 = 17.2930
            h3 = 0.0670
            if self.sigma1==1 and self.sigma2==0: # mode 1
                if self.p_memb > h1:
                    self.sigma1, self.sigma2 = 0, 1
            elif self.sigma1==0 and self.sigma2==1: # mode 2
                if self.p_memb < h2:
                    self.sigma1, self.sigma2 = 0, 0
            elif self.sigma1==0 and self.sigma2==0: # mode 3
                if self.p_memb < h3:
                    self.sigma1, self.sigma2 = 1, 0
                    self.e_Cbrine = self.inital_states['e_Cbrine']
                    self.e_Ck = self.inital_states['e_Ck']
            else:
                raise RuntimeError('Unknown Mode.')
        elif self.t == 'time':
            if (i-self.tmp)*self.step_len > 33:
                self.tmp = i
                if self.sigma1==1 and self.sigma2==0: # mode 1
                    self.sigma1, self.sigma2 = 0, 1
                elif self.sigma1==0 and self.sigma2==1: # mode 2
                    self.sigma1, self.sigma2 = 0, 0
                elif self.sigma1==0 and self.sigma2==0: # mode 3
                    self.sigma1, self.sigma2 = 1, 0
                    self.e_Cbrine = self.inital_states['e_Cbrine']
                    self.e_Ck = self.inital_states['e_Ck']
                else:
                    raise RuntimeError('Unknown Mode.')
        else:
            raise RuntimeError('Unknown Trans Mode.')

    def state_step(self, step_len):
        # e_RO20 in Chapter_13
        R_memb  = 0.202*(4.137e11*((self.e_Ck - 12000)/165 + 29))
        # e_RO1
        q_fp    = self.q_fp + step_len* \
                (- RO.R_fp*self.q_fp \
                 - self.p_tr \
                 + RO.p_fp*(1 - self.f_f)) \
                 /RO.I_fp
        # e_RO2
        p_tr    = self.p_tr + step_len* \
                (self.q_fp \
                 + self.sigma1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 - self.q_rp \
                 + self.sigma2*(self.p_memb - self.p_tr)/RO.R_return_s) \
                 /RO.C_tr
        # e_RO3
        d_q_rp  = step_len* \
                (- RO.R_rp*self.q_rp \
                 - RO.R_forward*self.q_rp \
                 - self.p_memb \
                 + RO.p_rp*(1 - self.f_r))/RO.I_rp
        q_rp    = (self.sigma1 + self.sigma2)*(self.q_rp + d_q_rp) \
                 + (1 - self.sigma1 - self.sigma2)*(self.p_tr - self.p_memb)/RO.R_forward
        # e_RO4
        p_memb  = self.p_memb + step_len* \
                (self.q_rp \
                 - self.p_memb/(R_memb*(1+self.f_m)) \
                 - self.sigma1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 - self.sigma2*(self.p_memb - self.p_tr)/RO.R_return_s \
                 - (1 - self.sigma1 - self.sigma2)*self.p_memb/RO.R_return_AES) \
                 / RO.C_memb
        # e_RO5
        e_Cbrine    = self.e_Cbrine + step_len*\
                (self.sigma1*(self.p_memb - self.p_tr)/RO.R_return_l \
                 + self.sigma2*(self.p_memb - self.p_tr)/RO.R_return_s \
                 + (1 - self.sigma1 - self.sigma2)*self.p_memb/RO.R_return_AES) \
                 / (1.667e-8*RO.C_brine)
        # e_RO6
        e_Ck    = self.e_Ck + step_len* \
                self.q_rp*(6*RO.C_brine + 0.1)/ (1.667e-8 * RO.C_k)
        
        # save & update
        self.states.append([q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck])
        self.para_faults.append([self.f_f, self.f_r, self.f_m])
        self.x.append((0 if not self.x else self.x[-1]) + step_len)
        self.q_fp, self.p_tr, self.q_rp, self.p_memb, self.e_Cbrine, self.e_Ck = q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck

    def state_step_matlab(self, step_len):
        # modes
        sigma_1 = self.sigma1
        sigma_2 = self.sigma2
        # states
        f_fp = self.q_fp
        p_tr = self.p_tr
        f_rp = self.q_rp
        p_mem = self.p_memb
        e_Cbrine = self.e_Cbrine
        e_Ck = self.e_Ck
        # inputs
        p_fp=RO.p_fp
        p_rp=RO.p_rp
        # sample rate
        T = step_len
        # parameters
        I_fp=RO.I_fp
        I_rp=RO.I_rp
        R_fp=RO.R_fp
        R_rp=RO.R_rp
        C_k=RO.C_k
        R_returns=RO.R_return_s
        R_returnl=RO.R_return_l
        R_returnASE=RO.R_return_AES
        R_forward=RO.R_forward
        C_tr=RO.C_tr
        C_memb=RO.C_memb
        C_brine=RO.C_brine
        # fault parameters
        f_f = self.f_f
        f_r = self.f_r
        f_m = self.f_m
        R_memb=0.202*(4.137*10**11*((e_Ck-12000)/165+29))

        df_fp =f_fp+ T*((1/I_fp)*(-R_fp*f_fp-p_tr+p_fp*(1-f_f)))
        dp_tr =p_tr+ T*((1/C_tr)*(f_fp+(sigma_1*(p_mem-p_tr)/R_returnl)-f_rp+(sigma_2*(p_mem-p_tr)/R_returns)))
        df_rp =(sigma_1+sigma_2)*(f_rp+ T*((1/I_rp)*(-R_rp*f_rp-R_forward*f_rp-p_mem+p_rp*(1-f_r))))+(1-sigma_1-sigma_2)*((p_tr-p_mem)/R_forward)
        dp_mem =p_mem+ T*((1/C_memb)*(f_rp+(p_mem/(R_memb*(1+f_m))-(sigma_1*(p_mem-p_tr)/R_returnl)-(sigma_2*(p_mem-p_tr)/R_returns))-(1-sigma_1-sigma_2)*(p_mem/R_returnASE)))
        de_Cbrine =e_Cbrine+ T*((1/(1.667*10**(-8)*C_brine))*((sigma_1*(p_mem-p_tr)/R_returnl)+(sigma_2*(p_mem-p_tr)/R_returns)+(1-sigma_1-sigma_2)*(p_mem/R_returnASE)))
        de_Ck =e_Ck+ T*((f_rp/C_k)*(6*C_brine+0.1)/(1.667*10**(-8)))

        return [df_fp, dp_tr, df_rp, dp_mem, de_Cbrine, de_Ck]

        # save & update
        # self.states.append([df_fp, dp_tr, df_rp, dp_mem, de_Cbrine, de_Ck])
        # self.para_faults.append([self.f_f, self.f_r, self.f_m])
        # self.x.append((0 if not self.x else self.x[-1]) + step_len)
        # self.q_fp, self.p_tr, self.q_rp, self.p_memb, self.e_Cbrine, self.e_Ck = df_fp, dp_tr, df_rp, dp_mem, de_Cbrine, de_Ck

    def np_modes(self):
        return np.array(self.modes)

    def np_states(self):
        return np.array(self.states)

    def np_obs(self):
        states = np.array(self.states)
        obs = states[:, (1, 3, 0, 4, 6)]
        return obs

    def np_para_faults(self):
        return np.array(self.para_faults)

    def _show(self, name):
        assert isinstance(name, str)
        if name in RO.states:
            i   = RO.states.index(name)
            y   = np.array(self.states)[:, i]
        elif name in RO.obs:
            name    = RO.obs[name]
            i   = RO.states.index(name)
            y   = np.array(self.states)[:, i]
        elif name=='modes':
            y   = np.array(self.modes)
        else:
            raise RuntimeError('Unknown name.')
        x = np.array(self.x)
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
        # fault parameters
        self.f_f    = 0
        self.f_r    = 0
        self.f_m    = 0
        # discrete modes
        self.sigma1 = None
        self.sigma2 = None
        # continous states
        self.init(self.inital_states)
        self.d_q_rp = 0
        # trajectory
        self.modes.clear()
        self.states.clear()
        self.para_faults.clear()
        self.x.clear()
