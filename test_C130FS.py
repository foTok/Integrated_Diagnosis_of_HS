'''
Test file for C130FS
'''
from Systems.C130FS import C130FS

c_t = input('Component type {valve, pump, tank}:')
c_i = int(input('Component id:'))
f_i = int(input('Fault time index:'))
f_t = int(input('Fault type:'))
f_m = float(input('Fault maganitude:'))

c130fs = C130FS()
c130fs.inject_fault(c_t, c_i, f_i, f_t, f_m)
c130fs.run()
c130fs.show_tanks()
c130fs.show_pumps()
c130fs.show_valves()
c130fs.show_balance()
print('DONE')
