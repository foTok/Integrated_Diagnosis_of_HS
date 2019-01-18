import pickle
import numpy as np

class term:
    def __init__(self, file_name, fault_type=None, fault_magnitude=None, fault_time=None):
        '''
        filename: str
        fault_type: None, int or str.
            int ~ discrete fault, the number is the id of fault mode
            str ~ parameter fault, the string is the name of the fault
        fault_magnitude: None or real
        fault_time: real
        '''
        self.file_name = file_name
        self.fault_type = fault_type
        self.fault_magnitude = fault_magnitude
        self.fault_time = fault_time

class cfg:
    def __init__(self, mode_names, state_names, output_names, variable_names, sample_int):
        '''
        filename: str
        mode_names: a list of str, the names of modes
        state_names: a list of str, the names of states
        output_names: a list of str, the names of outputs
        sample_int: real, sample interval
        '''
        self.mode_names = mode_names
        self.state_names = state_names
        self.output_names = output_names
        self.variable_names = variable_names
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
    def __init__(self, cfg_file):
        self.sample_int = 1.0
        self.cfg = pickle.load(cfg_file)
        self.data = []
        for term in self.cfg.terms:
            data_file = term.file_name
            data = np.load(data_file)
            self.data.append(data)

    def set_sample_int(self, si):
        # the new sample rate must be integer times the orginal one
        assert si/self.cfg.sample_int == int(si/self.cfg.sample_int)
        self.sample_int = si

    def add_noise(self, data, snr=None):
        if snr is None:
            return data
        ratio = 1/(10**(snr/20))
        std  = np.std(data, 0)
        noise = np.random.standard_normal(data.shape) * std * ratio
        data_with_noise = data + noise
        return data_with_noise
  
    def select_states(self, index):
        '''
        select all the states of the index_th file
        '''
        data = self.data[index]
        state_index = [self.cfg.variable_names.index(name) for name in self.cfg.state_names]
        states = data[:,state_index]
        x = np.arange(0, len(states), int(self.sample_int/self.cfg.sample_int))
        states = states[x, :]
        return states

    def select_outputs(self, index, snr=None):
        '''
        select all the outputs of the index_th file
        '''
        data = self.data[index]
        output_index = [self.cfg.variable_names.index(name) for name in self.cfg.output_names]
        outputs = data[:,output_index]
        x = np.arange(0, len(outputs), int(self.sample_int/self.cfg.sample_int))
        outputs = outputs[x, :]
        # add noise
        outputs_with_noise = self.add_noise(outputs, snr)
        return outputs_with_noise

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
