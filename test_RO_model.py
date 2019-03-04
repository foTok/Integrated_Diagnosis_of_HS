import argparse
import numpy as np
from RO import RO
from data_manager import data_manager
from utilities import obtain_var

if __name__ == "__main__":
    si = 0.001
    # show something
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample_int", type=float,  help="the sample interval")
    parser.add_argument("-f", "--fault", type=str, choices=['s_mode1', 's_mode2', 's_mode3', 'f_f', 'f_r', 'f_m'], help="the fault type")
    parser.add_argument("-m", "--magnitude", type=float, help="fault magnitude")
    parser.add_argument("-t", "--fault_time", type=float, help="fault time")
    parser.add_argument("-l", "--length", type=float, help="simulation length")
    args = parser.parse_args()

    init_state = [0, 0, 0, 0, 0, 0]
    sample_int = args.sample_int if args.sample_int is not None else 0.01
    fault_type = args.fault
    length = args.length if args.length is not None else 300

    ro = RO(sample_int) # 0.01 is the maximal available sample interval
    ro.run(0, init_state, length, fault_type, args.fault_time, args.magnitude)
    ro.plot_states()
    ro.plot_modes()

print('Done')
