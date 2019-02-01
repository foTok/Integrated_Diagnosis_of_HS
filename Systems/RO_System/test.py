import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,parentdir)
import argparse
import numpy as np
from RO import RO
from Systems.data_manager import data_manager
from utilities.utilities import obtain_var

if __name__ == "__main__":
    si = 0.001
    process_snr = 1000
    data_cfg = parentdir + '\\Systems\\RO_System\\data\\debug\\0.cfg'
    data_mana = data_manager(data_cfg, si)
    state = data_mana.select_states(0)
    pv = obtain_var(state, process_snr)
    # show something
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--sample_int", type=float,  help="the sample interval")
    parser.add_argument("-f", "--fault", type=str, choices=['stuck_1', 'stuck_2', 'stuck_3', 'f_f', 'f_r', 'f_m'], help="the fault type")
    parser.add_argument("-m", "--magnitude", type=float, help="fault magnitude")
    parser.add_argument("-t", "--fault_time", type=float, help="fault time")
    parser.add_argument("-l", "--length", type=float, help="simulation length")
    args = parser.parse_args()

    states  = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    init_state = [0, 0, 0, 0, 0, 0]
    sample_int = args.sample_int if args.sample_int is not None else 0.001
    fault_type = args.fault
    length = args.length if args.length is not None else 300

    ro = RO(sample_int) # 0.01 is the maximal available sample interval
    ro.set_state_disturb(pv)
    ro.run(init_state, length, fault_type, args.fault_time, args.magnitude)
    ro.plot_states()
    ro.plot_modes()


    # init_state = [0, 0, 0, 0, 0, 0]
    # ro1 = RO(0.01)
    # ro2 = RO(0.001)
    # ro1.run(init_state, 300)
    # ro2.run(init_state, 300)

    # states1 = ro1.np_states()
    # states2 = ro2.np_states()

    # x2 = np.arange(10, len(states2)+1, 10)-1
    # states2 = states2[x2,:]


print('Done')
