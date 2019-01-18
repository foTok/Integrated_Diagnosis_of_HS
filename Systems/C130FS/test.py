'''
Simulation file for C130FS
'''
import os
import sys
rootpath = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,rootpath)
from Systems.C130FS.C130FS import C130FS

c_t = input('Component type {valve, pump, tank}:')
if c_t != 'no':
    c_i = int(input('Component id:'))
    f_i = int(input('Fault time index:'))
    f_t = int(input('Fault type:'))
    f_m = float(input('Fault maganitude:'))

c130fs = C130FS()
if c_t!='no':
    c130fs.inject_fault(c_t, c_i, f_i, f_t, f_m)
c130fs.run()
c130fs.show_tanks()
c130fs.show_pumps()
c130fs.show_valves()
c130fs.show_balance()
print('DONE')
