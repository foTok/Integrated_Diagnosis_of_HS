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
from enum import Enum

mode_v = Enum('mode_v', ('open', 's_open', 'close', 's_close'))
mode_p = Enum('mode_p', ('open', 'close', 'failure'))

class C130FS:
    '''
    The class to simulate C130 fuel system behavior.
    '''
    def __init__(self):
        self._reset()
        
    def _reset(self):
        '''
        Reset all parameters.
        '''
        ################## Parameter Basis ##################
        # This means that the given parameter is based on \
        ###### *************** 10 min ************** ########
        _basis = 60*10 # (second)
        self._balance_begin = 0.15 # 15%
        self._balance_end   = 0.05 # 5%
        self._pump_open_line = 10  # 10 pounds
        self._pump_close_line= 0.5 # 0.5 pounds
        ###########System Normal Continuous Parameters#####
        # The resistances for 6 valves
        # The parameter from the slides are 4 which means if
        # we don't consider the change of pressure, the fuel
        # will reduce 1/4 every 10 min (by the default basis).
        # So each second, only 1/(4*_basis) fule goes out.
        self._R = np.array([4]*6) * _basis
        # fuel demand per second
        # The parameters are from the slides for 10 min.
        # If we need 60 pounds in 10 min, only 60/_basis pounds
        # are needed per second.
        self._demand = np.array([60, 40, 50, 50])/ _basis

        ################ System Fault Parameters ############
        # Partial stuck fault for valves. 0 by default.
        # Fault interval: [40, inf)
        # The fault parameters will BE effected by _basis.
        # But the default values are zero or inf, no need
        # to add them. We will use _basis when inject a fault.
        self._f_v = [0]*6
        # Leakage fault for tanks. inf by default.
        # Fault interval: [100, 200]
        self._f_t = [float('inf')]*6
        # No feed engouh fuel fault for pumps. 0 by default.
        # Fault interval: [0.1, 0.3]
        # The fault parameters will NOT be effected by _basis.
        self._f_p = [0]*6
        
        ########### System Discrete Mode Parameters #########
        # Inner mode for valves
        # 0 ~ close/off, 1 ~ open/on. close/off by default.
        self._sigma_v = [0]*6
        # Inner mode for pumps
        # 0 ~ closed/off, 1 ~ open/on.
        # pump 1~4 are in open by default.
        # pump 5, 6 are in close by default.
        self._sigma_p = [1]*6
        self._sigma_p[4] = 0
        self._sigma_p[5] = 0
        ########### System Continuous State Parameters ######
        # Pounds of fuel in tanks. Full by default
        # NOT effected by _basis
        # tank capacity
        self._tank_cap = [1340, 1230, 1230, 1340, 900, 900]
        # tank state
        self._tank = [1340, 1230, 1230, 1340, 900, 900]
        
        ################# System Outer Modes ################
        # Some components have outer discrete modes
        # For valves
        # A valve has four outer modes in all.
        # open, s_open ~ sigma = 1; close, s_close ~ sigma = 0
        self._mode_v = [mode_v.close] * 6
        # For pumps
        # A pump has three outer modes in all.
        # open ~ sigma = 1; close, failure ~ sigma = 0
        self._mode_p = [mode_p.open] * 6
        self._mode_p[4] = mode_p.close
        self._mode_p[5] = mode_p.close
        ########### Simulation and Sampling Parameters ######
        # Basic time unit is second
        self._sim_rate = 1
        self._smp_rate = 10

    def reset_fault(self):
        '''
        Reset all fault parameters.
        '''
        self._f_v = [0]*6
        self._f_t = [float('inf')]*6
        self._f_p = [0]*4
        self._f_pl = 0
        self._f_pr = 0

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

    def set_sim_smp(self, sim, smp):
        '''
        Set simulation and sampling rate.
        @para sim, an integer
        @para smp, an integer
        '''
        assert isinstance(sim, int)
        assert isinstance(smp, int)
        assert (smp/sim) == int(smp/sim)
        self._sim_rate = sim
        self._smp_rate = smp

    # For all valves
    def _valve_control(self, br, f, c_mode):
        '''
        return the next mode based on current mode
        @para br, balance rate, a non negative float or list
        @para f, discrete fault id, 0~no fault, 1~s_open, 2~s_close
        @pare c_mode, current mode
        @return n_mode, next mode
        '''
        # br ==> np.array([0.12]) or np.array([0.12, 0.13])
        if isinstance(br, float):
            br = np.array([br])
        else:
            br = np.array(br)
        # mode not change by default
        n_mode = c_mode
        if c_mode == mode_v.open:
            if f == 2:
                n_mode = mode_v.s_close
            else:
                if (br < self._balance_end).all() and f==0:
                    n_mode = mode_v.close
                elif (br < self._balance_end).all() and f==1:
                    n_mode = mode_v.s_open
        elif c_mode == mode_v.close:
            if f == 1:
                n_mode = mode_v.s_open
            else:
                if (br > self._balance_begin).any() and f==0:
                    n_mode = mode_v.open
                elif (br > self._balance_begin).any() and f==2:
                    n_mode = mode_v.s_close
        return n_mode

    # For pump 1~4
    def _pump_control(self, t, f, c_mode):
        '''
        Return the next mode and expected normal mode based on current mode
        @para t, the fuel amount in the tank
        @para f, fault id, 0~no fault, 1~failure
        @return n_mode, next mode
        @return e_mode, next expected mode if there is no fault
        '''
        #TODO: what is the e_mode when current mode is failure
        # mode not change by default
        n_mode = c_mode
        e_mode = c_mode
        if c_mode == mode_p.open:
            # If the next mode is close, we can not differentiate \
            # close and failure. So failure is not allowed to be \
            # injected here.
            if t < self._pump_close_line:
                e_mode = mode_p.close
                n_mode = mode_p.close
            else:# If it should keep open but f==1, the real mode is failure
                if f==1:
                    n_mode = mode_p.failure
        elif c_mode == mode_p.close:
            # If mode keeps close, we can not differenticate close \
            # and failure.So failure is not allowed to be injected \
            # here (The 'else' brach is ignored). If the expected \
            # mode is open, but f==1, the real mode should be failure
            if t > self._pump_open_line:
                e_mode = mode_p.open
                if f==0:
                    n_mode = mode_p.open
                elif f==1:
                    n_mode = mode_p.failure
        return n_mode, e_mode

    def _aux_pump_control(self, t, f, n_mp0, n_mp1, c_mode):
        '''
        Return the next mode of an auxiliary pump.
        @para t, the fuel amount in the tank
        @para f, fault id, 0~no fault, 1~failure
        @para n_mp0, the next mode of pump 0
        @para n_mp1, the next mode of pump 1
        @para c_mode, the current mode of the auxiliary tank
        @return n_mode, the next mode
        '''
        #TODO
        # mode not change by default
        n_mode = c_mode
        if c_mode == mode_p.open:
            pass
        elif c_mode == mode_p.close:
            pass
        return n_mode

    def _controler(self):
        '''
        Set the current modes based on the states
        '''
        # Balance rates
        br0 = abs(self._tank[0] - self._tank[1]) / min(self._tank_cap[0], self._tank_cap[1])
        br1 = abs(self._tank[2] - self._tank[3]) / min(self._tank_cap[2], self._tank_cap[3])
        br2 = abs(self._tank[0] - self._tank[3]) / min(self._tank_cap[0], self._tank_cap[3])
        br3 = abs(self._tank[1] - self._tank[2]) / min(self._tank_cap[1], self._tank_cap[2])
        br4 = abs(self._tank[4] - self._tank[5]) / min(self._tank_cap[4], self._tank_cap[5])
        # For valve 1
        if self._mode_v[0] == mode_v.open:
            pass
        elif self._mode_v[0] == mode_v.close:
            pass
        elif self._mode_v[0] == mode_v.s_open:
            pass
        else: # mode_v.s_close
            pass

    def _step(self):
        '''
        Forward one time step based on current modes and states
        '''
        pass
