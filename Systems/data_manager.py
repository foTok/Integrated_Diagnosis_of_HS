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
    def __init__(self, mode_names, state_names, output_names, sample_int):
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
