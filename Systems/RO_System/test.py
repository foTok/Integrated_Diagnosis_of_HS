import argparse
import numpy as np
from RO import RO


if __name__ == "__main__":
    # # show something
    # parser = argparse.ArgumentParser()
    # parser.add_argument("-s", "--sample_int", type=float,  help="the sample interval")
    # parser.add_argument("-f", "--fault", type=str, choices=['s0', 's1', 's2', 'f_f', 'f_r', 'f_m'], help="the fault type")
    # parser.add_argument("-m", "--magnitude", type=float, help="fault magnitude")
    # parser.add_argument("-t", "--fault_time", type=float, help="fault time")
    # parser.add_argument("-l", "--length", type=float, help="simulation length")
    # args = parser.parse_args()

    # faults = {'s0':3, 's1':4, 's2':5, 'f_f':'f_f', 'f_r':'f_r', 'f_m':'f_m'}
    # states  = ['q_fp', 'p_tr', 'q_rp', 'p_memb', 'e_Cbrine', 'e_Ck']
    # init_state = [0, 0, 0, 0, 0, 0]
    # sample_int = args.sample_int if args.sample_int is not None else 0.001
    # fault_type = faults[args.fault] if args.fault is not None else None
    # length = args.length if args.length is not None else 400

    # ro = RO(sample_int) # 0.01 is the maximal available sample interval 
    # ro.run(init_state, length, fault_type, args.fault_time, args.magnitude)
    # for s in states:
    #     ro.show(s)
    # ro.show('mode')


    init_state = [0, 0, 0, 0, 0, 0]
    ro1 = RO(0.01)
    ro2 = RO(0.001)
    ro1.run(init_state, 300)
    ro2.run(init_state, 300)

    states1 = ro1.np_states()
    states2 = ro2.np_states()

    x2 = np.arange(10, len(states2)+1, 10)-1
    states2 = states2[x2,:]


print('Done')
