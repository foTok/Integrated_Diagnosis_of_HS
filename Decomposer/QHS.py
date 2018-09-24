'''
This file describes Hybrid Systems in a Qualitative way,
so that we can decompose it.

MSOs Refs:
1.  The algorithm is described in 'Krysander, M., Åslund, J., & Nyberg, M. (2008). \
    An efficient algorithm for finding minimal overconstrained subsystems for model-based diagnosis. \
    IEEE Transactions on Systems, Man, and Cybernetics Part A:Systems and Humans, 38(1), 197–206. \
    https://doi.org/10.1109/TSMCA.2007.909555'

2.  The used DM decomposition algorithm is written in C by Timothy Davis from\
    'https://people.sc.fsu.edu/~jburkardt/c_src/csparse/csparse.html'

3.  The python wrapper code if written by xialulee \
    from 'http://blog.sina.com.cn/s/blog_4513dde60100o6m6.html'

'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
from Decomposer.PySparse import dmperm

class qmodel:
    '''
    Qualitive model:
        x1 = x2 + sigma*x3 + f1==> e1: x1, x2, x3, sigma, f1
    '''
    def __init__(self, warning=False):
        #The names of variables, a list
        self._variables = []
        #The attributes of variables, a list
        # 0 ~ unknown variables, by defaults
        # 1 ~ normal mode variables
        # 2 ~ discrete mode variables
        # 3 ~ fault parameter variables
        self._attributes = []
        #Qualitative equations
        # e1: (x1, x2, x3, sigma, f1)
        self._qequations = {}
        # if warning is True, warning information will be printed
        self._warning = warning

#######################################################################
##############Add variables and qeuations to define the model##########
#######################################################################
    def add_a_variable(self, name, attri):
        '''
        Add a variable.
        If it exists, IGNORE
        @para name, a string
        @para attri, {0, 1, 2, 3}
        '''
        if name not in self._variables:
            self._variables.append(name)
            self._attributes.append(attri)
        elif self._warning:
            print('{} is already in, IGNORE this time.', name)

    def add_variables(self, names, attris):
        '''
        Add variables
        @para names, a list of strings
        @para attris, a list of {0, 1, 2, 3}
        '''
        for name, attri in zip(names, attris):
            self.add_a_variable(name, attri)

    def vid(self, name):
        '''
        Find the id of variable "name"
        If not exits, return -1
        @para name, a string
        @return  -1 or a nature number
        '''
        if name not in self._variables:
            return -1
        return self._variables.index(name)

    def add_an_qequation(self, name, qequation):
        '''
        Add an equation. If name is in, OVERWRITE
        @para name, a string
        @para qequation, a tuple of strings which represent
            the names of included variables
        '''
        if (name in self._qequations) and self._warning:
            print('{} is already in, OVERWRITE this time.', name)
        self._qequations[name] = qequation

    def add_qequations(self, names, qequations):
        '''
        Add qequations
        @para names, a list of strings
        @para qequations, a list of qequations
        '''
        for name, qequation in zip(names, qequations):
            self.add_an_qequation(name, qequation)
    
#######################################################################
##############                      MSOs                     ##########
#######################################################################
    def MSOs(self):
        '''
        Return all the MSOs belongs to the qmodel
        '''
        pass

#######################################################################
##############                      BIPs                     ##########
#######################################################################
