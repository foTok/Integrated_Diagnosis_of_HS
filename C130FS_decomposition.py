'''
Decomposition file for C130FS
'''
from Decomposer.QHS import QModel

m = QModel()
# Add unknown variables
m.add_variables(['R', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', \
                 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', \
                 't1', 't2', 't3', 't4', 't5', 't6'], 'un',\
                 [False, False, False, False, False, False, False, \
                  False, False, False, False, False, False, \
                  True, True, True, True, True, True])

# Add normal mode variables
m.add_variables(['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', \
                 'sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6'], 'nm')

# Add fault parameter variables
m.add_variables(['fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', \
                 'fp1', 'fp2', 'fp3', 'fp4', 'fp5', 'fp6', \
                 'ft1', 'ft2', 'ft3', 'ft4', 'ft5', 'ft6'],'pf')

m.add_qequations(['ev0', 'ev1', 'ev2', 'ev3', 'ev4', 'ev5', 'ev6', \
                  'ep1', 'ep2', 'ep3', 'ep4', 'ep5', 'ep6', \
                  'et1', 'et2', 'et3', 'et4', 'et5', 'et6', \
                  'eo1', 'eo2', 'eo3', 'eo4', 'eo5', 'eo6'],
                [['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v1', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v2', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v3', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v4', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v5', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 'R', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6', 'v6', 't1', 't2', 't3', 't4', 't5', 't6'],
                 ['sp1', 'fp1', 'p1'],
                 ['sp2', 'fp2', 'p2'],
                 ['sp3', 'fp3', 'p3'],
                 ['sp4', 'fp4', 'p4'],
                 ['sp1', 'sp2', 'sp5', 'fp5', 'p5'],
                 ['sp3', 'sp4', 'sp6', 'fp6', 'p6'],
                 ['t1', 'v1', 'p1', 'ft1'],
                 ['t2', 'v2', 'p2', 'ft2'],
                 ['t3', 'v3', 'p3', 'ft3'],
                 ['t4', 'v4', 'p4', 'ft4'],
                 ['t5', 'v5', 'p5', 'ft5'],
                 ['t6', 'v6', 'p6', 'ft6'],
                 ['t1'],
                 ['t2'],
                 ['t3'],
                 ['t4'],
                 ['t5'],
                 ['t6']])

MSO = m.MSOs()

C   = [m.cost_and_pfaults(i) for i in MSO]

print('DONE')
