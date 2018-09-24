'''
Test file
'''
from Decomposer.QHS import QModel


m = QModel()
m.add_variables(['x1', 'x2', 'x3', 'x4'], 'un')
m.add_qequations(['e1', 'e2', 'e3', 'e4', 'e5', 'e6', 'e7', 'e8'],
                [['x1', 'x2'],
                 ['x2', 'x3'],
                 ['x3', 'x4'],
                 ['x1', 'x4'],
                 ['x1', 'x3'],
                 ['x2', 'x4'],
                 ['x1', 'x3'],
                 ['x2', 'x4']])

MSO = m.MSOs()
n = len(MSO)
r = False
for i in range(n):
    for j in range(i+1, n):
        r = r or (MSO[i]==MSO[j])
print("r =", r)
