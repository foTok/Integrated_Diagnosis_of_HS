'''
Some fault detectors.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
from scipy.stats import norm

def Z_test(res, window1, window2, conf=0.99):
    '''
    |window1|window2|
    if the length of res is less that window1+window2, return 0 directly.
    res: a list or np.array
    '''
    res = np.array(res)
    if len(res) < window1+window2:
        if len(res.shape)==1:
            return 0
        else:
            _, n = res.shape
            return np.array([0]*n)
    # mean and variance of window1
    res1 = res[-(window1+window2):-window2]
    mean1, var1 = np.mean(res1, 0), np.var(res1, 0)
    # mean of window2
    res2 = res[-window2:]
    mean2 = np.mean(res2)
    # the var should be modified based on windows
    abs_normalized = abs(mean2 - mean1) / np.sqrt(var1/window2)
    thresh = norm.ppf(1-(1-conf)/2)
    results = (abs_normalized > thresh) + 0
    return results
