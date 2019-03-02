'''
the class to extract information from the log
'''
import argparse
import numpy as np

class info:
    def __init__(self):
        self.fault_type = {}
    
    def add(self, fault_type, fault_time, fault_magnitude, \
            detect_delay, estimated_magnitude, mode_accuracy, state_error):
        if fault_type in self.fault_type:
            the_type_info = self.fault_type[fault_type]
        else:
            the_type_info = type_info()
            self.fault_type[fault_type] = the_type_info
        the_type_info.add(fault_time, fault_magnitude, \
                          detect_delay, estimated_magnitude, mode_accuracy, state_error)
    
    def print(self):
        msg_list = []
        for fault_type in self.fault_type:
            msg  = 'fault_type={}'.format(fault_type)
            msg_list.append(msg)
            print(msg)
            msg = self.fault_type[fault_type].print('\t')
            msg_list += msg
        return msg_list

class type_info:
    def __init__(self):
        self.fault_time = {}

    def add(self, fault_time, fault_magnitude, \
            detect_delay, estimated_magnitude, mode_accuracy, state_error):
        if fault_time in self.fault_time:
            the_time_info = self.fault_time[fault_time]
        else:
            the_time_info = time_info()
            self.fault_time[fault_time] = the_time_info
        the_time_info.add(fault_magnitude, \
                          detect_delay, estimated_magnitude, mode_accuracy, state_error)

    def print(self, prefix=''):
        msg_list = []
        for fault_time in self.fault_time:
            msg = '{}fault_time={}'.format(prefix, fault_time)
            msg_list.append(msg)
            print(msg)
            msg = self.fault_time[fault_time].print(prefix+'\t')
            msg_list += msg
        return msg_list

class time_info:
    def __init__(self):
        self.fault_magnitude = {}

    def add(self, fault_magnitude, \
            detect_delay, estimated_magnitude, mode_accuracy, state_error):
        if fault_magnitude in self.fault_magnitude:
            the_fault_info = self.fault_magnitude[fault_magnitude]
        else:
            the_fault_info = fault_info()
            self.fault_magnitude[fault_magnitude] = the_fault_info
        the_fault_info.add(detect_delay, estimated_magnitude, mode_accuracy, state_error)

    def print(self, prefix=''):
        msg_list = []
        for fault_magnitude in self.fault_magnitude:
            msg = '{}fault_magnitude={}'.format(prefix, fault_magnitude)
            msg_list.append(msg)
            print(msg)
            msg = self.fault_magnitude[fault_magnitude].print(prefix+'\t')
            msg_list += msg
        return msg_list

class fault_info:
    def __init__(self):
        self.detect_delay = []
        self.estimated_magnitude = []
        self.mode_accuracy = []
        self.state_error = []

    def add(self, detect_delay, estimated_magnitude, mode_accuracy, state_error):
        self.detect_delay.append(detect_delay)
        self.estimated_magnitude.append(estimated_magnitude)
        self.mode_accuracy.append(mode_accuracy)
        self.state_error.append(state_error)

    def print(self, prefix=''):
        detect_delay = '{}detect_delay={}, mu={}, sigma={}'\
        .format(prefix+'\t', np.round(self.detect_delay, 2), np.round(np.mean(self.detect_delay), 4), np.round(np.std(self.detect_delay), 4))
        estimated_magnitude = '{}estimated_magnitude={}, mu={}, sigma={}'\
        .format(prefix+'\t', np.round(self.estimated_magnitude, 4), np.round(np.mean(self.estimated_magnitude), 4), np.round(np.std(self.estimated_magnitude), 4))
        mode_accuracy = '{}mode_accuracy={}, mu={}, sigma={}'\
        .format(prefix+'\t', np.round(self.mode_accuracy, 4), np.round(np.mean(self.mode_accuracy), 4), np.round(np.std(self.mode_accuracy), 4))
        state_error = '{}state_error={}, mu={}, sigma={}'\
        .format(prefix+'\t', np.round(self.state_error, 4), np.round(np.mean(self.state_error), 4), np.round(np.std(self.state_error), 4))
        print(detect_delay)
        print(estimated_magnitude)
        print(mode_accuracy)
        print(state_error)
        return [detect_delay, estimated_magnitude, mode_accuracy, state_error]

def analyze_info(log_file_name, save_file_name):
    the_info = info()
    with open(log_file_name, 'r') as f:
        lines = f.readlines()
        fault_type, fault_time, fault_magnitude, detect_delay, estimated_magnitude, mode_accuracy, state_error = \
        None, None, None, None, None, None, None
        for line in lines:
            line=line.strip('\n')
            if line.endswith('***'): # reset line
                fault_type, fault_time, fault_magnitude, detect_delay, estimated_magnitude, mode_accuracy, state_error = \
                None, None, None, None, None, None, None
            elif line.endswith('s.'): # the line containing raw fault information
                line = line.split()
                fault_type = line[1]
                fault_magnitude = float(line[4])
                fault_time = float(line[-1][:-2])
            elif 'occurred' in line: # detect time and estimated magnitude
                line = line.split()
                detect_delay = float(line[4][:-2]) - fault_time
                estimated_magnitude = float(line[13][4:])+float(line[14])+float(line[15])
            elif 'accuracy' in line: # mode accuracy
                line = line.split()
                mode_accuracy = float(line[-1][:-1])
            elif 'n_mu' in line: # state error
                line = line.split()
                state_error = (float(line[4][1:])+float(line[5])+float(line[6])+float(line[7])+float(line[8])+float(line[9][:2]))/6
                the_info.add(fault_type, fault_time, fault_magnitude, \
                             detect_delay, estimated_magnitude, mode_accuracy, state_error)
            else:
                pass # do nothing
    msg = the_info.print()
    with open(save_file_name, 'w') as f:
        f.write('\n'.join(msg))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log', type=str, help='log file name')
    parser.add_argument('-o', '--out', type=str, help='output file name')
    args = parser.parse_args()

    analyze_info(args.log, args.out)
