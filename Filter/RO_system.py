from particle_fiter import hs_system

class RO(hs_system):
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
    def __init__(self, sample_int, process_var, obs_var):
        super(RO, self).__init__(['normal', 'pressure', 'reverse', 's_normal', 's_pressure', 's_reverse'], \
                                 ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck'], \
                                 sample_int, process_var, obs_var)
        # fault parameters
        self.f_f    = 0
        self.f_r    = 0
        self.f_m    = 0

    def set_parameter_fault(self, name, value):
        if name=='f_f':
            self.f_f = value
        elif name=='f_r':
            self.f_r = value
        elif name=='f_m':
            self.f_m = value
        else:
            pass # do nothing

    def modes(self, modes_i, states_i):
        h1 = 28.6770
        h2 = 17.2930
        h3 = 0.0670
        modes_ip1 = modes_i
        p = states_i[3]
        if modes_i == 0:
            if p > h1:
                modes_ip1 = 1
        if modes_i == 1:
            if p < h2:
                modes_ip1 = 2
        if modes_i == 2:
            if p < h3:
                modes_ip1 = 0
        else:
            pass
        return modes_ip1

    def states(self, modes_ip1, states_i):
        if (modes_ip1 % 3) == 0:
            _sigma1, _sigma2 = 1, 0
        elif (modes_ip1 % 3) == 1:
            _sigma1, _sigma2 = 0, 1
        elif (modes_ip1 % 3) == 2:
            _sigma1, _sigma2 = 0, 0
        else:
            pass # never
        # extract former state
        _q_fp, _p_tr, _q_rp, _p_memb, _e_Cbrine, _e_Ck = states_i
        # step forward
        step_len = self.si
        # e_RO20 in Chapter_13
        R_memb  = 0.202*(4.137e11*((_e_Ck - 12000)/165 + 29))
        # e_RO1
        q_fp    = _q_fp + step_len* \
                (- RO.R_fp*_q_fp \
                 - _p_tr \
                 + RO.p_fp*(1 - self.f_f)) \
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
                 + RO.p_rp*(1 - self.f_r))/RO.I_rp
        q_rp    = (_sigma1 + _sigma2)*(_q_rp + d_q_rp) \
                 + (1 - _sigma1 - _sigma2)*(_p_tr - _p_memb)/RO.R_forward
        # e_RO4
        p_memb  = _p_memb + step_len* \
                (_q_rp \
                 - _p_memb/(R_memb*(1 + self.f_m)) \
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
        states_ip1 = q_fp, p_tr, q_rp, p_memb, e_Cbrine, e_Ck
        output_ip1 = q_fp, p_tr, q_rp, e_Cbrine, e_Ck
        return states_ip1, output_ip1
