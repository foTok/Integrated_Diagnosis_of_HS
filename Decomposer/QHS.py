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
from Decomposer.utilities import M_plus
from Decomposer.utilities import belong_to

class QModel:
    '''
    Qualitive Model:
        x1 = x2 + sigma*x3 + f1==> e1: x1, x2, x3, sigma, f1
    '''
    def __init__(self, warning=False):
        #The names of variables, a list
        self._unknown_variables     = []
        self._normal_mode_variables = []
        self._fault_mode_variables  = []
        self._para_fault_variables  = []
        #All variables
        self._variables             = []
        #Qualitative equations
        # e1: (x1, x2, x3, sigma, f1)
        self._qequations = {}
        # if warning is True, warning information will be printed
        self._warning = warning

#######################################################################
##############Add variables and qeuations to define the model##########
#######################################################################
    def add_an_unknown_variable(self, name):
        '''
        Add an unknown variable.
        If it exists, IGNORE
        @para name, a string
        '''
        if name not in self._variables:
            self._unknown_variables.append(name)
            self._variables.append(name)
        elif self._warning:
            print('{} is already in, IGNORE this time.', name)

    def add_a_nmode_variable(self, name):
        '''
        Add a normal mode variable
        If it exists, IGNORE
        @para name, a string
        '''
        if name not in self._variables:
            self._normal_mode_variables.append(name)
            self._variables.append(name)
        elif self._warning:
            print('{} is already in, IGNORE this time.', name)

    def add_a_fmode_variable(self, name):
        '''
        Add a fault mode variable
        If it exists, IGNORE
        @para name, a string
        '''
        if name not in self._variables:
            self._fault_mode_variables.append(name)
            self._variables.append(name)
        elif self._warning:
            print('{} is already in, IGNORE this time.', name)

    def add_a_pfault_variable(self, name):
        '''
        Add a parameter fault variable
        If it exists, IGNORE
        @para name, a string
        '''
        if name not in self._variables:
            self._para_fault_variables.append(name)
            self._variables.append(name)
        elif self._warning:
            print('{} is already in, IGNORE this time.', name)

    def add_variables(self, names, vtype):
        '''
        Add unknown variables
        @para names, a list of strings
        @para vtype, {'un', 'nm', 'fm', 'pf'}
        '''
        if vtype == 'un':
            for name in names:
                self.add_an_unknown_variable(name)
        elif vtype == 'nm':
            for name in names:
                self.add_a_nmode_variable(name)
        elif vtype == 'fm':
            for name in names:
                self.add_a_fmode_variable(name)
        elif vtype == 'pf':
            for name in names:
                self.add_a_pfault_variable(name)
        elif self._warning:
            print('Unknown variable type: {}, IGNORE this time.', vtype)

    def add_an_qequation(self, name, qequation):
        '''
        Add an equation. If name is in, OVERWRITE
        @para name, a string
        @para qequation, a tuple of strings which represent \
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
        M = [{i} for i in self._qequations.keys()]
        MSO = self._find_MSO(M, M)
        return MSO

    def _unknown_matrix(self, names=None):
        '''
        Return a matrix to indicate the unknown variables in each qequation
        @para names, a set or list of interested qequation names. If names is None, \
                names should be all qequations in the QModel.
        @return umatrix, a n_row × n_col matrix where umatrix[i, j] = 1 means \
                that qequation i has the j_th unknown variable
        @return names, a list, names[i] is the name of the i_th qequation
        '''
        #Obtain interested qequations
        if names is None:
            names = list(self._qequations.keys())
        #To avoid that the modification of names will change the input set
        else:
            names = list(names)
        #Allocate space
        n_row = len(names)
        n_col = len(self._unknown_variables)
        umatrix = [[0]*n_col for _ in range(n_row)]
        #Construct matrix
        for r, n in zip(range(n_row), names):
            for c in range(n_col):
                if self._unknown_variables[c] in self._qequations[n]:
                    umatrix[r][c] = 1
        return umatrix, names

    def _lump(self, E, M_a):
        '''
        Lump variables
        Relationship R: e' not belong to (M\\{e})+
        @para E, a set of qequations
            {'e1', 'e2'}
        @para M_a, a list of qequation sets
            [{'e1'}, {'e2'},...]
        @return E_b a list of qequation sets, they are equivalence class
        '''
        E_b = [E]
        M_a = set.union(*M_a)
        #Remove any qequation in E_b from M_a
        s = E_b[0]
        for e in s:
            assert e in M_a
            M_a.remove(e)
        #Find the M+ part of M_a
        umatrix, names = self._unknown_matrix(M_a)
        M_p = M_plus(umatrix, names)
        M_0m = M_a - M_p  #M0 and M-
        for e in M_0m:
            E_b.append({e})
        return E_b

    def _unknown_redundancy(self, M):
        '''
        Compute the unknown redundancy of some qequations
        @para M, a list of qequation sets
        @return the number of redundancy
        '''
        M = set.union(*M)
        unknown_vars = set()
        for e in M:
            for v in self._unknown_variables:
                if v in self._qequations[e]:
                    unknown_vars.add(v)
        return len(M) - len(unknown_vars)

    def _find_MSO(self, M, R):
        '''
        Subroutine for MSOs
        @para M, a list of set
        @para R, a list of set to remove
        @return a list of MSOs
        Copy R so that the outer R will not be changed.
        '''
        if self._unknown_redundancy(M) == 1:
            return [set.union(*M)]
        R_a = []
        M_a = M
        R = R.copy()
        while len(R) > 0:
            E = R[0]
            E_b = self._lump(E, M_a)
            if belong_to(E_b, R):
                #Because E_b is always different, we can just append
                uE_b = set.union(*E_b)
                R_a.append(uE_b)
            for s in E_b:
                if s in R:
                    R.remove(s)
        MSO = []
        while len(R_a) > 0:
            E = R_a.pop()
            sM_a = M_a.copy()
            for e in E:
                sM_a.remove({e})
            sub_MSO = self._find_MSO(sM_a, R_a)
            MSO = MSO + sub_MSO
        return MSO

#######################################################################
##############                      BIPs                     ##########
#######################################################################



