'''
This file is used to simulate a simplified C130 fuel system.

The fuel system is composed of 6 valves, 6 tanks and 6 pumps. \
The tanks store fuel used by four engines. \
The pumps feed the fuel in the tanks to the engines. \
And the valves are employed to keep the whole system balanced. \
So, the control system has two objects:
•	Feed the engines fuel
•	Keep the system balanced
For the first purpose, TANK #1, TANK #2 and LH AUX TANK are used \
for the left two engines: ENG #1 and ENG #2. TANK #3, TANK #4 and \
RH AUX TANK are used for the right two engines. If TANK #1 and \
TANK #2 have enough fuel, PUMP #1 and PUMP #2 extract fuel in them \
and feed ENG #1 and ENG #2 respectively. If one of them has no \
enough fuel, LH AUX PUMP will use fuel in LH AUX TANK to feed \
ENG #1 or ENG #2, or both. Similar cases for the right part.

Specifications
	Fuel Demands:
	engine#1=60 pounds per 10 min
	engine#2=40 pounds per 10 min
	engine#3=50 pounds per 10 min⁡
	engine#4=50 pounds per 10 min
	Acceptable pressure differences between the tanks:
	    Max(tank1,tank2)< 20%
	    Max(tank3,tank4)<20"%" 
	    Max(tank1,tank4)<20"%" 
	    Max(tank2,tank3)<20"%" 
	    Max(tankLA,tankRA)<20"%" 
	When there is enough fuel in each tank:
	    Tank1                        engine#1
	    Tank2                        engine#2
	    Tank3                        engine#3
	    Tank4                        engine#4
	When there is not enough fuel in one of the main tanks use the \
        auxiliary tank in that side to feed the engines in that side. 
        To make sure that the engines get enough fuel, we assume that \
        when the fuel height or pressure is low enough (small but not zero), \
        the corresponding pump will be closed, and the auxiliary pump will be open.
        When the thresholds are violated, some valve are opened to balance them.
        We assume the control logic is the following:
    When the pressure difference is above 15%, the valves attached to \
        the tanks will be open. When the pressure difference is below 5%, \
        the values will be closed.

'''
import numpy as np
import matplotlib.pyplot as plt

class C130FS:
    '''
    The class to simulate C130 fuel system behavior.
    '''
    def __init__(self, sim=1):
        ################## Parameter Basis ##################
        # This means that the given parameter is based on
        #       *************** 10 min **************
        # Basic unit for simr is second.
        # Please do not set simr as some weird number such as 0.14142515
        self.sim_rate = sim
        self.demand = np.array([60, 40, 50, 50]) / (60*10) * sim
        self.R = np.array([4]*6)*(60*10)/sim
        # control parameters
        self.balance_begin = 200
        self.balance_end = 50
        self.pump_line = 10
        
        self.state = [] # init [1340, 1230, 1230, 1340, 900, 900]
        self.mode = []
        # states: 1, 2, 3, 4, L, R
        # modes: pump 1,2,3,4,L,R + valve 1,2,3,4,L,R
        #        pump mode: close, open, failure => 0, 1, 2.
        #                   open ~ sigma = 1; close, failure ~ sigma = 0
        #        mode valve: close, open, s_close, s_open => 0, 1, 2, 3
        #                   close, s_close ~ sigma = 0; open, s_open => sigma = 1
        # fault parameters: pump 1,2,3,4,L,R + tank 1,2,3,4,L,R + valve 1,2,3,4,L,R

    def valve_i_mode_step(self, mode, br):
        br = np.array(br)
        if mode==0 and (br > self.balance_begin).any():
            mode = 1
        elif mode==1 and (br < self.balance_end).all():
            mode = 0
        else:
            pass # do nothing
        return mode

    def pump_i_mode_step(self, p, h): # pump 1~4
        if p==0 and h>self.pump_line:
            p = 1
        elif p==1 and h < self.pump_line:
            p = 0
        else:
            pass # do nothing
        return p

    def pump_mode_step(self, modes, states):
        p1, p2, p3, p4, pl, pr = modes
        h1, h2, h3, h4, hl, hr = states
        # pump 1 ~ 4
        p1 = self.pump_i_mode_step(p1, h1)
        p2 = self.pump_i_mode_step(p2, h2)
        p3 = self.pump_i_mode_step(p3, h3)
        p4 = self.pump_i_mode_step(p4, h4)
        # pump l
        if pl==0 and (h1<self.pump_line or h2<self.pump_line) and (hl>self.pump_line):
            pl = 1
        elif pl==1 and ((h1>self.pump_line and h2>self.pump_line) or hl<self.pump_line):
            pl = 0
        else:
            pass # do nothing
        # pump r
        if pr==0 and (h3<self.pump_line or h4<self.pump_line) and (hr>self.pump_line):
            pr = 1
        elif pr==1 and ((h3>self.pump_line and h4>self.pump_line) or hr<self.pump_line):
            pr = 0
        else:
            pass # do nothing
        return [p1, p2, p3, p4, pl, pr]


    # For all valves
    def valve_mode_step(self, modes, states):
        '''
        modes: The modes of valves.
        states: all states.
        '''
        v1, v2, v3, v4, vl, vr = modes
        h1, h2, h3, h4, hl, hr = states
        br0 = abs(h1 - h2)
        br1 = abs(h3 - h4)
        br2 = abs(h1 - h4)
        br3 = abs(h2 - h3)
        br4 = abs(hl - hr)
        # valve 1
        v1 = self.valve_i_mode_step(v1, [br0, br2])
        # valve 2
        v2 = self.valve_i_mode_step(v2, [br0, br3])
        # valve 3
        v3 = self.valve_i_mode_step(v3, [br1, br3])
        # valve 4
        v4 = self.valve_i_mode_step(v4, [br1, br2])
        # valve L
        vl = self.valve_i_mode_step(vl, [br4])
        # valve R
        vr = self.valve_i_mode_step(vr, [br4])
        return [v1, v2, v3, v4, vl, vr]

    def fault_parameters(self, t=None, fault_type=None, fault_time=None, fault_magnitude=None):
        return None, [0]*6 + [float('inf')]*6 + [0]*6

    def mode_step(self, mode_i, state_i):
        mode_p = mode_i[0:6]
        mode_v = mode_i[6:12]
        mode_p = self.pump_mode_step(mode_p, state_i)
        mode_v = self.valve_mode_step(mode_v, state_i)
        mode_ip1 = mode_p + mode_v
        return mode_ip1

    def state_step(self, mode_ip1, state_i, fault_parameters):
        '''
        Forward one time step based on current modes and states
        '''
        mode_ip1 = np.array(mode_ip1)
        state_i = np.array(state_i)
        fault_parameters = np.array(fault_parameters)
        sigma_p = mode_ip1[0:6]
        sigma_v = mode_ip1[6:12]
        tank = state_i
        f_p = fault_parameters[0:6]
        f_t = fault_parameters[6:12]
        f_v = fault_parameters[12:18]
        # R
        R_v = self.R + f_v
        T = 0
        if sum(sigma_v)!=0:
            R = 0
            for i in range(6):
                R = R + sigma_v[i]/R_v[i]
                T = T + sigma_v[i]*tank[i]/R_v[i]
            R = 1/R
            T = T*R
        # Valve i
        valve = [0]*6
        if T!=0:
            for i in range(6):
                valve[i] = sigma_v[i]*(T - tank[i])/R_v[i]
        # Pump i
        pump = [0]*6
        for i in range(4):# For pump1~pump4
            pump[i] = sigma_p[i] * self.demand[i] * (1 - f_p[i])
        # For the left auxiliary pump
        pump[-2] = sigma_p[-2]*\
                  ((1 - sigma_p[0])*self.demand[0] + (1 - sigma_p[1])*self.demand[1])*\
                  (1 - f_p[-2])
        # For the right auxiliary pump
        pump[-1] = sigma_p[-1]*\
                  ((1 - sigma_p[2])*self.demand[2] + (1 - sigma_p[3])*self.demand[3])*\
                  (1 - f_p[-1])
        # Tank i
        n_tank = [0]*6
        for i in range(6):
            n_tank[i] = tank[i] + valve[i] - pump[i] - tank[i] / f_t[i]
        return n_tank

    def run(self, init=[1340, 1230, 1230, 1340, 900, 900]):
        state = init if not self.state else self.state[-1]
        mode = ([1,1,1,1]+[0]*8) if not self.mode else self.mode[-1]
        while sum(mode[0:6])!=0:
            mode = self.mode_step(mode, state)
            _mode, fault_para = self.fault_parameters()
            mode = mode if _mode is None else _mode
            state = self.state_step(mode, state, fault_para)
            self.mode.append(mode)
            self.state.append(state)

    def np_state(self):
        return np.array(self.state)

    def np_mode(self):
        return np.array(self.mode)

    def show_tanks(self):
        '''
        Show the fuel in the tanks.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Fuel in tanks')  # Add a title so we know which it is
        data = self.np_state()
        ax_lst[0, 0].plot(data[:, 0]) # Tank 1
        ax_lst[1, 0].plot(data[:, 1]) # Tank 2
        ax_lst[2, 0].plot(data[:, 4]) # Tank left auxiliary
        ax_lst[0, 1].plot(data[:, 2]) # Tank 3
        ax_lst[1, 1].plot(data[:, 3]) # Tank 4
        ax_lst[2, 1].plot(data[:, 5]) # Tank right auxiliary
        plt.show()

    def show_pumps(self):
        '''
        Show the modes of pumps.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Pump inner modes')  # Add a title so we know which it is
        data = self.np_mode()
        ax_lst[0, 0].plot(data[:, 0])
        ax_lst[1, 0].plot(data[:, 1])
        ax_lst[2, 0].plot(data[:, 4])
        ax_lst[0, 1].plot(data[:, 2])
        ax_lst[1, 1].plot(data[:, 3])
        ax_lst[2, 1].plot(data[:, 5])
        plt.show()

    def show_valves(self):
        '''
        Show the modes of valves.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('valve inner modes')  # Add a title so we know which it is
        data = self.np_mode()
        ax_lst[0, 0].plot(data[:, 6])
        ax_lst[1, 0].plot(data[:, 7])
        ax_lst[2, 0].plot(data[:, 8])
        ax_lst[0, 1].plot(data[:, 9])
        ax_lst[1, 1].plot(data[:, 10])
        ax_lst[2, 1].plot(data[:, 11])
        plt.show()
