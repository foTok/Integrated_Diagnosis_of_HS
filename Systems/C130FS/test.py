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
c130fs.run()
states = c130fs.np_states()
c130fs.plot_states()
c130fs.plot_modes()
print('DONE')
