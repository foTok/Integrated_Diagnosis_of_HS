'''
Decomposition file for C130FS
'''
from Decomposer.QHS import QModel

m = QModel()
# Add unknown variables
m.add_variables(['T', 'v1', 'v2', 'v3', 'v4', 'v5', 'v6', \
                 'p1', 'p2', 'p3', 'p4', 'p5', 'p6', \
                 't1', 't2', 't3', 't4', 't5', 't6', \
                 't1_', 't2_', 't3_', 't4_', 't5_', 't6_'], 'un',\
                 [False, False, False, False, False, False, False, \
                  False, False, False, False, False, False, \
                  True, True, True, True, True, True, \
                  False, False, False, False, False, False])

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
                  'eo1', 'eo2', 'eo3', 'eo4', 'eo5', 'eo6', \
                  'eo1_', 'eo2_', 'eo3_', 'eo4_', 'eo5_', 'eo6_'],
                [['T', 'sv1', 'sv2', 'sv3', 'sv4', 'sv5', 'sv6', 't1_', 't2_', 't3_', 't4_', 't5_', 't6_', 'fv1', 'fv2', 'fv3', 'fv4', 'fv5', 'fv6'],
                 ['T', 'sv1', 'v1', 'fv1'],
                 ['T', 'sv2', 'v2', 'fv2'],
                 ['T', 'sv3', 'v3', 'fv3'],
                 ['T', 'sv4', 'v4', 'fv4'],
                 ['T', 'sv5', 'v5', 'fv5'],
                 ['T', 'sv6', 'v6', 'fv6'],
                 ['sp1', 'fp1', 'p1'],
                 ['sp2', 'fp2', 'p2'],
                 ['sp3', 'fp3', 'p3'],
                 ['sp4', 'fp4', 'p4'],
                 ['sp1', 'sp2', 'sp5', 'fp5', 'p5'],
                 ['sp3', 'sp4', 'sp6', 'fp6', 'p6'],
                 ['t1', 't1_', 'v1', 'p1', 'ft1'],
                 ['t2', 't2_', 'v2', 'p2', 'ft2'],
                 ['t3', 't3_', 'v3', 'p3', 'ft3'],
                 ['t4', 't4_', 'v4', 'p4', 'ft4'],
                 ['t5', 't5_', 'v5', 'p5', 'ft5'],
                 ['t6', 't6_', 'v6', 'p6', 'ft6'],
                 ['t1'],
                 ['t2'],
                 ['t3'],
                 ['t4'],
                 ['t5'],
                 ['t6'],
                 ['t1_'],
                 ['t2_'],
                 ['t3_'],
                 ['t4_'],
                 ['t5_'],
                 ['t6_']])

MSO = m.MSOs()

C   = [m.cost_and_pfaults(i) for i in MSO]

print('DONE')
