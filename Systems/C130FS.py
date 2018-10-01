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
from enum import Enum

mode_v = Enum('mode_v', ('open', 's_open', 'close', 's_close'))
mode_p = Enum('mode_p', ('open', 'close', 'failure'))

class C130FS:
    '''
    The class to simulate C130 fuel system behavior.
    '''
    def __init__(self, sim=1):
        self._reset(sim)
        
    def _reset(self, sim):
        '''
        Reset all parameters.
        @para simr, simulation rate
        '''
        ################## Parameter Basis ##################
        # This means that the given parameter is based on
        #       *************** 10 min **************
        # Basic unit for simr is second.
        # Please do not set simr as some weird number such as 0.14142515
        assert sim > 0
        self._sim_rate = sim
        self._basis = 60*10 / sim  # (second) / simr
        self._br_type = 'per2c' #percentage of minimal capacity(per2c), percentage of the current average fuel(per2a), gallons(gal)
        self._balance_begin = 0.10 * (1 if self._br_type!='gal' else 100)
        self._balance_end   = 0.05 * (1 if self._br_type!='gal' else 100)
        self._pump_open_line = 10
        self._pump_close_line= 0.5
        ###########System Normal Continuous Parameters#####
        # The resistances for 6 valves
        # The parameter from the slides are 4 which means if
        # we don't consider the change of pressure, the fuel
        # will reduce 1/4 every 10 min (by the default basis).
        # So each second, only 1/(4*_basis) fule goes out.
        self._R = np.array([2]*6) * self._basis
        # fuel demand per second
        # The parameters are from the slides for 10 min.
        # If we need 60 pounds in 10 min, only 60/_basis pounds
        # are needed per second.
        self._demand = np.array([60, 40, 50, 50])/ self._basis

        ################ System Fault Parameters ############
        # Partial stuck fault for valves. 0 by default.
        # Fault interval: [40, inf)
        # The fault parameters will BE effected by _basis.
        # But the default values are zero or inf, no need
        # to add them. We will use _basis when inject a fault.
        self._f_v = [0]*6
        self._f_vb = [0]*6
        # Leakage fault for tanks. inf by default.
        # Fault interval: [100, 200]. BE effected by _basis
        self._f_t = [float('inf')]*6
        self._f_tb = [float('inf')]*6
        # No feed engouh fuel fault for pumps. 0 by default.
        # Fault interval: [0.1, 0.3]
        # The fault parameters will NOT be effected by _basis.
        self._f_p = [0]*6
        self._f_pb = [0]*6
        
        ########### System Discrete Mode Parameters #########
        # Inner mode for valves
        # 0 ~ close/off, 1 ~ open/on. close/off by default.
        self._sigma_v = [[0]*6]
        # Inner mode for pumps
        # 0 ~ closed/off, 1 ~ open/on.
        # pump 1~4 are in open by default.
        # pump 5, 6 are in close by default.
        self._sigma_p = [[1]*6]
        self._sigma_p[0][4] = 0
        self._sigma_p[0][5] = 0
        ########### System Continuous State Parameters ######
        # Pounds of fuel in tanks. Full by default
        # NOT effected by _basis
        # tank capacity
        self._tank_cap = [1340, 1230, 1230, 1340, 900, 900]
        # tank state
        self._tank = [[1240, 1230, 1230, 1240, 900, 900]]
        
        ################# System Outer Modes ################
        # Some components have outer discrete modes
        # For valves
        # A valve has four outer modes in all.
        # open, s_open ~ sigma = 1; close, s_close ~ sigma = 0
        self._mode_v = [[mode_v.close] * 6]
        # For pumps
        # A pump has three outer modes in all.
        # open ~ sigma = 1; close, failure ~ sigma = 0
        self._mode_p = [[mode_p.open] * 6]
        self._mode_p[0][4] = mode_p.close
        self._mode_p[0][5] = mode_p.close
        # The expected modes for the first 4 pumps
        self._e_mode_p = [[mode_p.open] * 4]
        ################### Sampling Parameters #############
        # Basic time unit is second
        self._smp_rate = sim
        #################### Time Steps #####################
        self._i = 0
        #################### Time Steps #####################
        self._balance = []
        ############## Fault Injection Information ##########
        # The fault time step when try to inject a fault.
        # Warning: it may not be the real inject time. For example \
        # if the current mode is close for a pump, the injection is \
        # invalid. Failure will be injected at the first time step \
        # the injection is allowed.
        self._fi = float('inf')
        # Fault flag
        # The first 6 entries are for valves, the next 6 entries \
        # are for pumps, and the last 6 entries are for tanks.
        # _f2[0]~_f2[4] ~ (v1~v4), _f2[4] ~ v_left, _f2[5] ~ v_right
        # _f2[6]~_f2[10] ~ (p1~p4), _f2[10] ~ p_aux_left, _f2[11] ~ p_aux_right
        # _f2[12]~_f2[16] ~ (t1~t4), _f2[16] ~ t_aux_left, _f2[17] ~ t_aux_right
        # For a valve: 0~normal, 1~s_open, 2~s_close, 3~p_stuck
        # For a pump: 0~normal, 1~failure, 2~leakage
        # For a tank: 0~normal, 1~leakage
        self._f2 = [0] * 18
        self._f2b= [0] * 18

    def reset_fault(self):
        '''
        Reset all fault parameters.
        '''
        self._f_v = [0]*6
        self._f_t = [float('inf')]*6
        self._f_p = [0]*6
        self._fi   = float('inf')
        self._f2   = [0] * 18

    def set_states(self, states):
        '''
        Set the current states
        @para state, a list
        '''
        assert len(states) == 6
        self._tank[:] = states[:]

    def set_modes(self, modes):
        '''
        Set the current modes
        @para modes, a list
        '''
        assert len(modes) == 12
        self._sigma_v[:] = modes[0:6]
        self._sigma_p[:] = modes[6:12]

    def set_sim_smp(self, smp):
        '''
        Set sampling rate.
        smp will be adjusted to integer times of _sim_rate.
        @para smp, an integer
        '''
        assert isinstance(smp, int)
        smp = int(smp / self._sim_rate) * self._sim_rate
        self._smp_rate = smp

    def inject_fault(self, c_t, c_i, f_i, f_t, f_m=None):
        ''''
        Inject a fault.
        @para c_t, component type, {'valve', 'pump', 'tank'}
        @para c_i, component id, {1, 2, 3, 4, 'left', 'right'}
        @para f_i, fault time step
        @para f_t, fault type
        @para f_m, fault magnitude
        '''
        assert f_i > 0
        self._fi = f_i
        # Obtain index
        if isinstance(c_i, int):
            index = c_i - 1
        elif c_i == 'left':
            index = 4
        elif c_i == 'right':
            index = 5
        else:
            raise TypeError('Unknown component id')

        if c_t == 'valve':
            assert f_t==1 or f_t==2 or f_t==3
            self._f2b[0+index] = f_t
            if f_t==3:
                assert f_m is not None
                self._f_vb[index] = f_m * self._basis
        elif c_t == 'pump':
            assert f_t==1 or f_t==2
            self._f2b[6+index] = f_t
            if f_t==2:
                assert f_m is not None
                assert 0< f_m < 1
                self._f_pb[index] = f_m
        elif c_t == 'tank':
            assert f_t==1
            assert f_m is not None
            self._f2b[12+index] = f_t
            self._f_tb[index] = f_m * self._basis
        else:
            raise TypeError('Unknown fault type')

    def _try2active_pf(self):
        '''
        Try to active a parameter fault.
        '''
        self._f_v[:] = self._f_vb[:]
        self._f_p[:] = self._f_pb[:]
        self._f_t[:] = self._f_tb[:]

    # For all valves
    def _valve_control(self, br, f, c_mode):
        '''
        return the next mode based on current mode
        @para br, balance rate, a non negative float or list
        @para f, discrete fault id, 0~no fault, 1~s_open, 2~s_close
        @pare c_mode, current mode
        @return n_mode, next mode
        '''
        assert f==0 or f==1 or f==2 or f==3
        # For parameter fault, _valve_control ignore it.
        if f==3:
            f = 0
        # br ==> np.array([0.12]) or np.array([0.12, 0.13])
        if isinstance(br, float):
            br = np.array([br])
        else:
            br = np.array(br)
        # Mode not change by default
        n_mode = c_mode
        # When c_mode is a fault mode, because of 'n_mode=c_mode', 
        # we can ignore the else branch.
        if c_mode == mode_v.open:
            if (br < self._balance_end).all():
                # 1. Should be close in theory.
                # 2. Because we cannot differentiate close and s_close, 
                # s_close (f==2) is not allowed to be injected here. Just keep close.
                # 3. If inject stuck open, the mode should be s_open
                n_mode = mode_v.close if f!=1 else mode_v.s_open
            else:
                # In else branch, n_mode should be open in theory.
                # s_open should not be injected here, but s_close might
                n_mode = mode_v.open if f!=2 else mode_v.s_close
        elif c_mode == mode_v.close:
            if (br > self._balance_begin).any():
                # 1. Should be in open in theory.
                # 2. We cannot differentiate open and s_open. So, s_open should
                # NOT be injected here, but s_close can.
                n_mode = mode_v.open if f!=2 else mode_v.s_close
            else:
                # 1. Should be in close in theory.
                # 2. We cannot differentiate close and s_close. So, s_close should
                # NOT be injected here, but s_open can.
                n_mode = mode_v.close if f!=1 else mode_v.s_open
        return n_mode

    # For pump 1~4
    def _pump_control(self, t, f, c_mode):
        '''
        Return the next mode based on the current mode
        @para t, the fuel amount in the tank
        @para f, fault id, 0~no fault, 1~failure
        @return n_mode, next mode
        '''
        assert f==0 or f==1 or f==2
        # For parameter fault, ignore here.
        if f==2:
            f = 0
        # Mode not change by default
        n_mode = c_mode
        # When c_mode is a fault mode, because of 'n_mode=c_mode', 
        # we can ignore the else branch.
        if c_mode == mode_p.open:
            # If the next mode is close, we can not differentiate \
            # close and failure. So failure is not allowed to be \
            # injected here.
            if t < self._pump_close_line:
                n_mode = mode_p.close
            else:# If it should keep open but f==1, the real mode is failure
                if f==1:
                    n_mode = mode_p.failure
        elif c_mode == mode_p.close:
            # If mode keeps close, we can not differenticate close \
            # and failure. So failure is not allowed to be injected \
            # here (The 'else' brach is ignored). If the expected \
            # mode is open, but f==1, the real mode should be failure
            if t > self._pump_open_line:
                n_mode = mode_p.open if f!=1 else mode_p.failure
        return n_mode
    
    # For pump 1~4
    def _pump_expect_control(self, t, c_e_mode):
        '''
        Return the next expected mode based on the current expected mode
        @para t, the fuel amount in the tank
        @return e_mode, next expected mode assuming there is no fault
        '''
        # Mode not change by default
        n_e_mode = c_e_mode
        if c_e_mode == mode_p.open:
            if t < self._pump_close_line:
                n_e_mode = mode_p.close
        elif c_e_mode == mode_p.close:
            if t > self._pump_open_line:
                n_e_mode = mode_p.open
        else:
            raise TypeError('Unknown expected control mode.')
        return n_e_mode

    def _aux_pump_control(self, t, f, n_e_mp0, n_e_mp1, c_mode):
        '''
        Return the next mode of an auxiliary pump.
        @para t, the fuel amount in the tank
        @para f, fault id, 0~no fault, 1~failure
        @para n_e_mp0, the next expected mode of pump 0
        @para n_e_mp1, the next expected mode of pump 1
        @para c_mode, the current mode of the auxiliary tank
        @return n_mode, the next mode
        '''
        assert f==0 or f==1 or f==2
        # For parameter fault, ignore here
        if f==2:
            f = 0
        # Mode not change by default
        n_mode = c_mode
        if c_mode == mode_p.open:
            # If the next mode is close, we cannot differentiate \
            # close and failure. So failure is not allowed to be \
            # injected here.
            if t < self._pump_close_line or \
                (n_e_mp0 == mode_p.open and n_e_mp1 == mode_p.open):
                n_mode = mode_p.close
            else: # Expected n_mode should be open, unless we inject failure
                if f==1:
                    n_mode = mode_p.failure
        elif c_mode == mode_p.close:
            # In the if branch, n_mode should be open unless we inject failure. \
            # In the else branch, we cannot differentiate close and failure. \
            # So, failure cannot be injected. We just neglect the branch.
            if t > self._pump_open_line and \
                (n_e_mp0 == mode_p.close or n_e_mp1 == mode_p.close):
                n_mode = mode_p.open if f!=1 else mode_p.failure
        return n_mode

    def _sigma(self, o_mode):
        '''
        Return the sigma value based on the outer mode
        @para o_mode, outer mode, mode_v or mode_p
        @return sigma, inner mode
        '''
        if isinstance(o_mode, mode_v):
            if o_mode == mode_v.open or o_mode == mode_v.s_open:
                sigma = 1
            else: # mode_v.close or mode_v.s_close
                sigma = 0
        elif isinstance(o_mode, mode_p):
            if o_mode == mode_p.open:
                sigma = 1
            else:# mode_p.close or mode_p.failure
                sigma = 0
        else:
            raise TypeError('Unknown outer mode')
        return sigma

    def _sigmas(self, o_modes):
        '''
        Just like _sigma, but _sigmas will handle a list of o_mode
        @para o_modes, a list of o_mode
        @return sigmas, a list of sigma
        '''
        sigmas = []
        for o_mode in o_modes:
            sigma = self._sigma(o_mode)
            sigmas.append(sigma)
        return sigmas

    def _controller(self):
        '''
        Set the current modes based on the states
        '''
        # Inject a fault
        if self._i > self._fi:
            self._f2[:] = self._f2b[:]
            self._try2active_pf()
        # Balance rates
        tank = self._tank[self._i]
        if self._br_type == 'per2c':
            br0 = abs(tank[0] - tank[1]) / min(self._tank_cap[0], self._tank_cap[1])
            br1 = abs(tank[2] - tank[3]) / min(self._tank_cap[2], self._tank_cap[3])
            br2 = abs(tank[0] - tank[3]) / min(self._tank_cap[0], self._tank_cap[3])
            br3 = abs(tank[1] - tank[2]) / min(self._tank_cap[1], self._tank_cap[2])
            br4 = abs(tank[4] - tank[5]) / min(self._tank_cap[4], self._tank_cap[5])
        elif self._br_type == 'per2a':
            br0 = 2 * abs(tank[0] - tank[1]) / (tank[0] + tank[1])
            br1 = 2 * abs(tank[2] - tank[3]) / (tank[2] + tank[3])
            br2 = 2 * abs(tank[0] - tank[3]) / (tank[0] + tank[3])
            br3 = 2 * abs(tank[1] - tank[2]) / (tank[1] + tank[2])
            br4 = 2 * abs(tank[4] - tank[5]) / (tank[4] + tank[5])
        elif self._br_type == 'gal':
            br0 = abs(tank[0] - tank[1])
            br1 = abs(tank[2] - tank[3])
            br2 = abs(tank[0] - tank[3])
            br3 = abs(tank[1] - tank[2])
            br4 = abs(tank[4] - tank[5])
        else:
            raise TypeError('Unknown balance rate')
        self._balance.append([br0, br1, br2, br3, br4])
        ################################ For valves ###############################
        c_mode_v = self._mode_v[self._i]
        n_mode_v = []
        # For valve 1
        n_mode = self._valve_control([br0, br2], self._f2[0], c_mode_v[0])
        n_mode_v.append(n_mode)
        # For valve 2
        n_mode = self._valve_control([br0, br3], self._f2[1], c_mode_v[1])
        n_mode_v.append(n_mode)
        # For valve 3
        n_mode = self._valve_control([br1, br3], self._f2[2], c_mode_v[2])
        n_mode_v.append(n_mode)
        # For valve 4
        n_mode = self._valve_control([br1, br2], self._f2[3], c_mode_v[3])
        n_mode_v.append(n_mode)
        # For left auxiliary valve
        n_mode = self._valve_control(br4, self._f2[4], c_mode_v[4])
        n_mode_v.append(n_mode)
        # For right auxiliary valve
        n_mode = self._valve_control(br4, self._f2[5], c_mode_v[5])
        n_mode_v.append(n_mode)
        # Add n_mode_v into self._mode_v
        self._mode_v.append(n_mode_v)
        ################################ For pumps ###############################
        c_mode_p = self._mode_p[self._i]
        e_mode_p = self._e_mode_p[self._i]
        n_mode_p = []
        n_e_mode_p = []
        # For pump 1
        n_mode = self._pump_control(tank[0], self._f2[6+0], c_mode_p[0])
        n_e_mode = self._pump_expect_control(tank[0], e_mode_p[0])
        n_mode_p.append(n_mode)
        n_e_mode_p.append(n_e_mode)
        # For pump 2
        n_mode = self._pump_control(tank[1], self._f2[6+1], c_mode_p[1])
        n_e_mode = self._pump_expect_control(tank[1], e_mode_p[1])
        n_mode_p.append(n_mode)
        n_e_mode_p.append(n_e_mode)
        # For pump 3
        n_mode = self._pump_control(tank[2], self._f2[6+2], c_mode_p[2])
        n_e_mode = self._pump_expect_control(tank[2], e_mode_p[2])
        n_mode_p.append(n_mode)
        n_e_mode_p.append(n_e_mode)
        # For pump 4
        n_mode = self._pump_control(tank[3], self._f2[6+3], c_mode_p[3])
        n_e_mode = self._pump_expect_control(tank[3], e_mode_p[3])
        n_mode_p.append(n_mode)
        n_e_mode_p.append(n_e_mode)
        # For left auxiliary pump
        n_mode = self._aux_pump_control(tank[4], self._f2[6+4], n_e_mode_p[0], n_e_mode_p[1], c_mode_p[4])
        n_mode_p.append(n_mode)
        # For right auxiliary pump
        n_mode = self._aux_pump_control(tank[5], self._f2[6+5], n_e_mode_p[2], n_e_mode_p[3], c_mode_p[5])
        n_mode_p.append(n_mode)
        # Add into self._mode_p and self._e_mode_p
        self._mode_p.append(n_mode_p)
        self._e_mode_p.append(n_e_mode_p)
        ############### No Discrete Mode Control For pumps #####################
        self._i += 1

    def _step(self):
        '''
        Forward one time step based on current modes and states
        '''
        self._controller()  #self._i += 1 has been executed in self._controller()
        tank = self._tank[self._i - 1]
        n = len(tank)
        sigma_v = self._sigmas(self._mode_v[self._i])
        sigma_p = self._sigmas(self._mode_p[self._i])
        # Store them
        self._sigma_v.append(sigma_v)
        self._sigma_p.append(sigma_p)
        # R
        R_v = self._R + np.array(self._f_v)
        if sum(sigma_v)==0:
            R = float('inf')
        else:
            R = 0
            for i in range(n):
                R = R + (1/R_v[i])*sigma_v[i]
            R = 1/R
        # Valve i
        valve = [0]*n
        # In the else brach, valve[:]=0 by default. So, ignored.
        if R!=float('inf'):
            for i in range(n):
                # In the else branch, valve[i]=0 by default. So, ignored.
                if sigma_v[i]!=0:
                    for k in range(n):
                        valve[i] = valve[i] + ((tank[k]-tank[i])/R_v[k] if sigma_v[k]==1 else 0)
                    valve[i] = valve[i] * R / R_v[i]
        # Pump i
        pump = [0]*n
        for i in range(n-2):# For pump1~pump4
            pump[i] = sigma_p[i] * self._demand[i] * (1 - self._f_p[i])
        # For the left auxiliary pump
        pump[-2] = sigma_p[-2]*\
                  ((1 - sigma_p[0])*self._demand[0] + (1 - sigma_p[1])*self._demand[1])*\
                  (1 - self._f_p[-2])
        # For the right auxiliary pump
        pump[-1] = sigma_p[-1]*\
                  ((1 - sigma_p[2])*self._demand[2] + (1 - sigma_p[3])*self._demand[3])*\
                  (1 - self._f_p[-1])
        # Tank i
        n_tank = [0]*n
        for i in range(n):
            n_tank[i] = tank[i] + valve[i] - pump[i] - tank[i] / self._f_t[i]
        # Store new states
        self._tank.append(n_tank)
        # When all tanks are empty, stop simulation
        stop_flag = (sum(sigma_p)!=0)
        return stop_flag

    def run(self):
        '''
        Run the simulation
        '''
        flag = True
        while flag:
            flag = self._step()
        
    def show_tanks(self):
        '''
        Show the fuel in the tanks.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Fuel in tanks')  # Add a title so we know which it is
        data = np.array(self._tank)
        ax_lst[0, 0].plot(data[:, 0]) # Tank 1
        ax_lst[1, 0].plot(data[:, 1]) # Tank 2
        ax_lst[2, 0].plot(data[:, 4]) # Tank left auxiliary
        ax_lst[0, 1].plot(data[:, 2]) # Tank 3
        ax_lst[1, 1].plot(data[:, 3]) # Tank 4
        ax_lst[2, 1].plot(data[:, 5]) # Tank right auxiliary
        plt.show()

    def show_pumps(self):
        '''
        Show the fuel in the tanks.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Pump inner modes')  # Add a title so we know which it is
        data = np.array(self._sigma_p)
        ax_lst[0, 0].plot(data[:, 0]) # Tank 1
        ax_lst[1, 0].plot(data[:, 1]) # Tank 2
        ax_lst[2, 0].plot(data[:, 4]) # Tank left auxiliary
        ax_lst[0, 1].plot(data[:, 2]) # Tank 3
        ax_lst[1, 1].plot(data[:, 3]) # Tank 4
        ax_lst[2, 1].plot(data[:, 5]) # Tank right auxiliary
        plt.show()

    def show_valves(self):
        '''
        Show the fuel in the tanks.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Valve inner modes')  # Add a title so we know which it is
        data = np.array(self._sigma_v)
        ax_lst[0, 0].plot(data[:, 0]) # Tank 1
        ax_lst[1, 0].plot(data[:, 1]) # Tank 2
        ax_lst[2, 0].plot(data[:, 4]) # Tank left auxiliary
        ax_lst[0, 1].plot(data[:, 2]) # Tank 3
        ax_lst[1, 1].plot(data[:, 3]) # Tank 4
        ax_lst[2, 1].plot(data[:, 5]) # Tank right auxiliary
        plt.show()

    def show_balance(self):
        '''
        Show the fuel in the tanks.
        '''
        fig, ax_lst = plt.subplots(3, 2)  # A figure with a 2x3 grid of Axes
        fig.suptitle('Balance rates')  # Add a title so we know which it is
        data = np.array(self._balance)
        ax_lst[0, 0].plot(data[:, 0]) 
        ax_lst[1, 0].plot(data[:, 1]) 
        ax_lst[2, 0].plot(data[:, 2]) 
        ax_lst[0, 1].plot(data[:, 3])
        ax_lst[1, 1].plot(data[:, 4])
        plt.show()