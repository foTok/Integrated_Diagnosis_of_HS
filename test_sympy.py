'''
A file to learn how to use SymPy.
The file should be renamed and used to compute all the residuals later.
'''

import numpy as np
from sympy import *


_basis = 60*10
# Inner Paramters
_R = np.array([2]*6)*_basis
_D = np.array([60, 40, 50, 50])/_basis
# Unknown variables
T, v1, v2, v3, v4, v5, v6, \
p1, p2, p3, p4, p5, p6, \
t1, t2, t3, t4, t5, t6, \
t1_, t2_, t3_, t4_, t5_, t6_ = \
symbols('T, v1, v2, v3, v4, v5, v6, \
         p1, p2, p3, p4, p5, p6, \
         t1, t2, t3, t4, t5, t6, \
         t1_, t2_, t3_, t4_, t5_, t6_')

# Normal mode variables
sv1, sv2, sv3, sv4, sv5, sv6, \
sp1, sp2, sp3, sp4, sp5, sp6 = \
symbols('sv1, sv2, sv3, sv4, sv5, sv6, \
         sp1, sp2, sp3, sp4, sp5, sp6')

#TODO
# Equations
ev0 = Eq(1/(sv1/_R[0] + sv2/_R[1] +sv3/_R[2] +sv4/_R[3] +sv5/_R[4] +sv6/_R[5])* \
         (sv1*t1_/_R[0] + sv2*t2_/_R[1] + sv3*t3_/_R[2] + sv4*t4_/_R[3] + sv5*t5_/_R[4] + sv6*t6_/_R[5]), T)
ev1 = Eq((T-sv1*t1_)/_R[0], v1)
ev2 = Eq((T-sv2*t2_)/_R[1], v2)
ev3 = Eq((T-sv3*t3_)/_R[2], v3)
ev4 = Eq((T-sv4*t4_)/_R[3], v4)
ev5 = Eq((T-sv5*t5_)/_R[4], v5)
ev6 = Eq((T-sv6*t6_)/_R[5], v6)
ep1 = Eq(sp1*_D[0], p1)
ep2 = Eq(sp2*_D[1], p2)
ep3 = Eq(sp3*_D[2], p3)
ep4 = Eq(sp4*_D[3], p4)
ep5 = Eq(sp5*((1-sp1)*_D[0]+(1-sp2)*_D[1]), p5)
ep6 = Eq(sp6*((1-sp3)*_D[2]+(1-sp4)*_D[3]), p6)
et1 = Eq(t1_+v1-p1, t1)
et2 = Eq(t2_+v2-p2, t2)
et3 = Eq(t3_+v3-p3, t3)
et4 = Eq(t4_+v4-p4, t4)
et5 = Eq(t5_+v5-p5, t5)
et6 = Eq(t6_+v6-p6, t6)

r = solve([ev4, ev6, ep4, ep6, et4, et6], (t6))
print(r)
print('DONE')
