'''
This document implementes some particle filter algorithms.
'''
import numpy as np
from scipy.stats import chi2

def chi2_confidence(x, df):
    '''
    return the confidence
    x is the value which is normalized.
    df is the freedom
    '''
    return 1 - chi2.cdf(x, df)

def normalize(particles):
    w = 0
    for ptc in particles:
        w += ptc.weight
    for ptc in particles:
        ptc.weight /= w

class hybrid_particle:
    '''
    a hybrid particle contains both discrete modes and continuous states
    '''
    def __init__(self, mode_names, state_names, mode_values=None, state_values=None, weight=1):
        self.mode_names = mode_names        # list, not change
        self.state_names = state_names      # list, not change
        self.mode_values = mode_values      # int, may change
        self.state_values = state_values    # list or np.array, may change
        self.weigth = weight                # real, may change
        self.fault_type = None              # int or str
        self.fault_magnitude = None         # real

    def set_hs(self, mode_values, state_values):
        '''
        set hybrid states
        '''
        self.mode_values = mode_values
        self.state_values = state_values

    def set_mode(self, mode_values):
        '''
        set mode values
        '''
        self.mode_values = mode_values

    def set_state(self, state_values):
        '''
        set state values
        '''
        self.state_values = state_values

    def set_weigth(self, weight):
        self.weigth = weight

    def set_fault(self, fault_type, fault_magnitude=None):
        self.fault_type = fault_type
        self.fault_magnitude = fault_magnitude

    def clone(self):
        pct = hybrid_particle(self.mode_names,\
                              self.state_names,\
                              self.mode_values,\
                              self.state_values.copy(),\
                              self.weigth)
        return pct

class hs_system_wrapper:
    '''
    def the interface of hs_system for filter
    '''
    def __init__(self, hs=None, process_var=None, obs_var=None):
        self.hs = hs
        self.pv = process_var # np.array
        self.ov = obs_var     # np.array
        self.p_std = np.sqrt(process_var) # np.array

    def fault_parameters(self, fault_type, fault_magnitude):
        return self.hs.fault_parameters(0, fault_type, 1, fault_magnitude)

    def mode_step(self, mode_i, state_i):
        return self.hs.mode_step(mode_i, state_i)

    def state_step(self, mode_ip1, state_i, fault_parameters):
        return self.hs.state_step(mode_ip1, state_i, fault_parameters)

class chi2_hpf:
    def __init__(self, hsw=hs_system_wrapper()):
        # The default parameter of hsw has no significance
        # but help the text editor to find the right methods
        # so that programming could be easier.
        self.hsw = hsw # hs_system_wrapper
        self.tracjectory = []

    def init_particles(self, modes, state_mean, state_var, N):
        particles= []
        disturbance = np.random.randn(N, len(state_mean))
        disturbance *= np.sqrt(state_var)
        for i in range(N):
            states = state_mean + disturbance[i]
            ptc = hybrid_particle(self.hsw.mode_names,\
                                  self.hsw.state_names,\
                                  modes,\
                                  states,\
                                  1/N)
            particles.append(ptc)
        return particles

    def step(self, particles, obs):
        '''
        particles: hybrid_particle list
        '''
        particles_ip1 = []
        for ptc in particles:
            p = ptc.clone()
            # one step based on the partical
            modes, states = self.hsw.mode_step(p.mode_values, p.state_values)
            mode_fault, para_fault = self.hsw.fault_parameters(p.fault_type, p.fault_magnitude)
            modes = modes if mode_fault is None else mode_fault
            states, output = self.hsw.state_step(modes, states, para_fault)
   
            p.set_hs(modes, states)
            # compute Pobs
            res = (obs - output)**2/self.hsw.obs_var
            Pobs = chi2_confidence(np.sum(res), len(res))
            p.set_weigth(p.weight*Pobs)
            particles_ip1.append(p)
        normalize(particles_ip1)
        return particles_ip1

    def track(self, modes, state_mean, state_var, N, observations):
        for obs in observations:
            if self.tracjectory:
                particles = self.tracjectory[-1]
            else:
                particles = self.init_particles(modes, state_mean, state_var, N)
            particles_ip1 = self.step(particles, obs)
            self.tracjectory.append(particles_ip1)              
