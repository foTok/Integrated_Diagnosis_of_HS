'''
Simulation file for C130FS_new
'''
import os
import sys
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootpath)
import matplotlib.pyplot as plt
from Systems.C130FS.C130FS_new import C130FS

c130fs = C130FS(0.1)
c130fs.run()
state = c130fs.np_state()
c130fs.show_tanks()
c130fs.show_pumps()
c130fs.show_valves()
print('DONE')
