'''
Simulation file for C130FS_new
'''
import os
import sys
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootpath)
import matplotlib.pyplot as plt
from Systems.C130FS.C130FS import C130FS

c130fs = C130FS(0.1)
c130fs.run(init=[1340, 1230, 1230, 1340, 900, 900], t=0, fault_type='leak_1', fault_time=5000, fault_magnitude=0.05)
states = c130fs.np_states()
c130fs.plot_states()
c130fs.plot_modes()
print('DONE')
