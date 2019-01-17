'''
This file includes some utility functions to be used.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from Decomposer.PySparse import dmperm


def M_plus(umatrix, names):
    '''
    Return the M+ part of all the qeuqations
    @para umatrix, a 2-d list, row represents a qequation, col represents an unknown variable
    @para the names of all qequations
    @return a set
    '''
    p, _, _, _, _, rr = dmperm(umatrix)
    start = rr[2]
    M_plus_num = p[start:]
    M_plus = set()
    for i in M_plus_num:
        M_plus.add(names[i])
    return M_plus

def is_subset_of(s0, s1):
    '''
    Judge if s0 is the subset of s1
    @para s0 the subset if return True
    @para s1 the superset if return True
    @return True~s0 belongs to s1, False otherwise
    '''
    for e in s0:
        if e not in s1:
            return False
    return True
