'''
Try to decompose RO System.
'''
import os
import sys
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootpath)
from Decomposer.QHS import QModel

m = QModel()
# Add unknown variables, v_ is the derivative of v.
m.add_variables(['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck', 'R_memb', 'd_q_rp',\
                 'q_fp_', 'p_tr_', 'q_rp_', 'p_memb_', 'e_Cbrine_', 'e_Ck_', 'd_q_rp_'], 'un',\
                 [False, False, False, False, False, False, False, False,\
                  True, True, True, True, True, True, True])

# Add normal mode variables
m.add_variables(['sigma1', 'sigma2'], 'nm')

# Add fault parameter variables
m.add_variables(['f_f', 'f_r', 'f_m'],'pf')

m.add_qequations(['e0', 'e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7',\
                  'ei0','ei1','ei2','ei3','ei4','ei5','ei6',\
                  'eo0', 'eo1', 'eo2', 'eo3', 'eo4', 'eo5',\
                  'et0', 'et1'],
                [['R_memb', 'e_Ck'],
                 ['q_fp_', 'q_fp', 'p_tr', 'f_f'],
                 ['p_tr_', 'q_fp', 'p_memb', 'p_tr', 'q_rp', 'sigma1', 'sigma2'],
                 ['q_rp', 'd_q_rp', 'p_tr', 'p_memb', 'sigma1', 'sigma2'],
                 ['p_memb_', 'q_rp', 'R_memb','p_memb', 'p_tr', 'sigma1', 'sigma2', 'f_m'],
                 ['e_Cbrine_', 'p_memb', 'p_tr', 'sigma1', 'sigma2'],
                 ['e_Ck_', 'q_rp', 'e_Cbrine'],
                 ['d_q_rp_', 'q_rp', 'p_memb', 'f_r'],
                 ['q_fp', 'q_fp_'],
                 ['p_tr', 'p_tr_'],
                 ['q_rp', 'q_rp_'],
                 ['p_memb', 'p_memb_'],
                 ['e_Cbrine', 'e_Cbrine_'],
                 ['e_Ck', 'e_Ck_'],
                 ['d_q_rp', 'd_q_rp_'],
                 ['p_tr'],
                 ['p_memb'],
                 ['q_fp'],
                 ['e_Cbrine'],
                 ['e_Ck'],
                 ['q_rp'],
                 ['sigma1'],
                 ['sigma2']])

MSO = m.MSOs()

C   = [m.cost_MSO(i) for i in MSO]

MSOs = MSO[-6:]
C_i  = m.cost_isolate(MSOs)

print('DONE')
