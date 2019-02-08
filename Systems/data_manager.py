import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
import pickle
import torch
import random
import progressbar
import numpy as np
from utilities.utilities import add_noise

class term:
    def __init__(self, file_name, fault_type=None, fault_time=None, fault_magnitude=None):
        '''
        filename: str
        fault_type: None, int or str.
            int ~ discrete fault, the number is the id of fault mode
            str ~ parameter fault, the string is the name of the fault
        fault_magnitude: None or real
        fault_time: real
        '''
        self.file_name = os.path.basename(file_name)
        self.fault_type = fault_type
        self.fault_time = fault_time
        self.fault_magnitude = fault_magnitude

class cfg:
    def __init__(self, mode_names, state_names, output_names, variable_names, fault_para_names, labels, sample_int):
        '''
        filename: str
        mode_names: a list or dict of str, the names of modes
        state_names: a list of str, the names of states
        output_names: a list of str, the names of outputs
        sample_int: real, sample interval
        '''
        self.mode_names = mode_names
        self.state_names = state_names
        self.output_names = output_names
        self.variable_names = variable_names
        self.fault_para_names = fault_para_names
        self.labels = labels
        self.sample_int = sample_int
        self.terms = []

    def add_term(self, the_term):
        '''
        the_term: term
        '''
        self.terms.append(the_term)

    def save(self, file_name):
        if not file_name.endswith('.cfg'):
            file_name += '.cfg'
        with open(file_name, 'wb') as f:
            pickle.dump(self, f)

class data_manager:
    '''
    The class is used to manage data.
    There are two modes.
    Mode 1 is used to offer the whole data of one simulation so that
    we can test the filtering algorithms.
    Mode 2 is used to sample some data so that we can train and test 
    ANNs.
    '''
    def __init__(self, cfg_file, si=1.0):
        # load config file
        path = os.path.dirname(cfg_file)
        with open(cfg_file, 'rb') as f:
            cfg = pickle.load(f)
        self.cfg = cfg
        self.data = []
        self.labels = tuple(self.cfg.labels) # fix the order
        with progressbar.ProgressBar(max_value=100) as bar:
            print('Loading data...')
            term_len = len(self.cfg.terms)
            for i, term in enumerate(self.cfg.terms):
                data_file = os.path.join(path, term.file_name)
                data_file = data_file if data_file.endswith('.npy') else data_file+'.npy'
                data = np.load(data_file)
                self.data.append(data)
                bar.update(float('%.2f'%((i+1)*100/term_len)))
        # set sample step len
        assert si/self.cfg.sample_int == int(si/self.cfg.sample_int)
        self.sample_int = si

    def select_states(self, index, snr_or_pro=None, norm=None):
        '''
        select all the states of the index_th file
        '''
        data = self.data[index]
        state_index = [self.cfg.variable_names.index(name) for name in self.cfg.state_names]
        states = data[:,state_index]
        interval = int(self.sample_int/self.cfg.sample_int)
        x = np.arange(interval, len(states)+1, interval)-1
        states = states[x, :]
        states_with_noise = add_noise(states, snr_or_pro)
        if norm is not None:
            states_with_noise = states_with_noise / norm
        return states_with_noise

    def select_outputs(self, index, snr_or_pro=None, norm=None):
        '''
        select all the outputs of the index_th file
        '''
        data = self.data[index]
        output_index = [self.cfg.variable_names.index(name) for name in self.cfg.output_names]
        outputs = data[:,output_index]
        x = np.arange(0, len(outputs), int(self.sample_int/self.cfg.sample_int))
        x = x[1:]
        outputs = outputs[x, :]
        # add noise
        outputs_with_noise = add_noise(outputs, snr_or_pro)
        if norm is not None:
            outputs_with_noise = outputs_with_noise / norm
        return outputs_with_noise

    def select_modes(self, index):
        data = self.data[index]
        modes_index = [self.cfg.variable_names.index(name) for name in self.cfg.mode_names]
        modes = data[:,modes_index]
        x = np.arange(0, len(modes), int(self.sample_int/self.cfg.sample_int))
        x = x[1:]
        modes = modes[x, :]
        return modes

    def get_info(self, index, prt=True):
        '''
        get the information of the index_th file
        '''
        term = self.cfg.terms[index]
        fault_type = term.fault_type
        fault_magnitude = term.fault_magnitude
        fault_time = term.fault_time
        if prt:
            print('fault {} with magnitude {} at {}s.'\
                  .format(fault_type, fault_magnitude, fault_time))
        return fault_type, fault_magnitude, fault_time

    def get_labels(self):
        return self.labels

    def get_mode_size(self):
        mode_size = []
        for m in self.cfg.mode_names:
            mode_size.append(len(self.cfg.mode_names[m]))
        return mode_size

    def sample(self, size, window, limit, normal_proportion, snr_or_pro=None, norm_o=None, norm_s=None):
        '''
        size:
            int, the number of sampled data.
        window:
            int, the lenght of each data point.
        limit:
            a tuple with two ints, (n1, n2).
            n1 means we have to make sure there are at least n1 data come before the fault time.
            n2 is similar.
            n1+n2 < window
        *******both window and limit are set as second**********
        normal_proportion:
            float between 0 and 1, the proportion of the normal mode
        '''
        assert sum(limit) < window
        window = int(window/self.sample_int)
        limit = (int(limit[0]/self.sample_int), int(limit[1]/self.sample_int))
        label_size = len(self.labels) # if label_size is one, it must be normal
        assert label_size >= 1
        fault_size = 0 if label_size==1 else int(size*(1-normal_proportion)/(label_size-1))
        normal_size = int(size - fault_size*(len(self.labels) - 1)) # make sure it is an int
        # hs0: the initial hybrid states, the input of classifiers,
        # x: system outputs, the input of classifiers,
        # m: modes, the output of classifiers,
        # y: states, the output of classifiers, 
        # p: the fault parameters, the output of classifiers.
        hs0, x, m, y, p = [], [], [], [], []
        for label in self.labels:
            # 1. find all indexes with this label
            indexes = []
            for i, term in enumerate(self.cfg.terms):
                l = 'normal' if term.fault_type is None else str(term.fault_type)
                if l==label:
                    indexes.append(i)
            # 2. select one file randomly in iteration
            iter_size = normal_size if label=='normal' else fault_size
            for _ in range(iter_size):
                i = indexes[random.randint(0, len(indexes)-1)]
                term = self.cfg.terms[i]
                outputs_i = self.select_outputs(i, snr_or_pro=snr_or_pro, norm=norm_o)
                states_i = self.select_states(i, norm=norm_s)
                modes_i = self.select_modes(i)
            # 3. pick out data in a window
                if label=='normal':
                    l1, l2 = 0, len(outputs_i)-window
                else:
                    fault_i = int(term.fault_time / self.sample_int)
                    l1, l2 = fault_i + limit[1] - window, fault_i - limit[0]
                start = random.randint(l1, l2)
                _hs0 = np.concatenate((modes_i[start-1], states_i[start-1])) # hs0
                outputs = outputs_i[start:start+window, :] # x
                modes = modes_i[start+window, :] # m
                states = states_i[start+window, :] # y
                # fault parameters
                _p = np.zeros(len(self.cfg.fault_para_names))
                if term.fault_type in self.cfg.fault_para_names:
                    _p[self.cfg.fault_para_names.index(term.fault_type)] = term.fault_magnitude
                # store them
                hs0.append(_hs0)
                x.append(outputs)
                m.append(modes)
                y.append(states)
                p.append(_p)
        hs0, x, m, y, p = np.array(hs0), np.array(x), np.array(m), np.array(y), np.array(p)
        return hs0, x, m, y, p

    def np2target(self, y):
        '''
        args:
            y: batch × time × mode_num, np.array
        '''
        y_list = []
        mode_size = self.get_mode_size()
        batch, _ = y.shape
        for i, size in enumerate(mode_size):
            y_i = y[:, i]
            y_torch_i = torch.zeros((batch, size))
            for b in range(batch):
                k = int(y_i[b])
                y_torch_i[b, k] = 1
            y_list.append(y_torch_i)
        return y_list

    def np2paratarget(self, y):
        batch, num = y.shape
        base = np.arange(1, num+1)
        y_torch = torch.zeros((batch, num+1))
        for b in range(batch):
            para = y[b,:]
            i = (para!=0)*base
            y_torch[b,i] = 1
        return y_torch
