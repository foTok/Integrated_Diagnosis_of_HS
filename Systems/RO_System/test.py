import numpy as np
from RO_model import RO

states  = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
ro = RO(0.001)
ro.run(1000)
for s in states:
    ro.show(s)
print('DONE')
