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
        self.mode_values = mode_values      # list, may change
        self.state_values = state_values    # np.array, may change
        self.weigth = weight                # real, may change

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

    def clone(self):
        pct = hybrid_particle(self.mode_names,\
                              self.state_names,\
                              self.mode_values[:],\
                              self.state_values.copy(),\
                              self.weigth)
        return pct

class hs_system:
    '''
    def the interface of hs_system for filter
    '''
    def __init__(self, mode_names, state_names, sample_int, process_var, obs_var):
        self.mode_names = mode_names
        self.state_names = state_names
        self.si = sample_int  # sample interval, real
        self.pv = process_var # np.array
        self.ov = obs_var     # np.array
        self.p_std = np.sqrt(process_var) # np.array

    def set_parameter_fault(self, name, value):
        print('You must implement this interface')

    def modes(self, modes_i, states_i):
        print('You must implement this interface')
        modes_ip1 = None
        return modes_ip1

    def states(self, modes_ip1, states_i):
        print('You must implement this interface')
        states_ip1 = None
        output_ip1 = None
        return states_ip1, output_ip1

class chi2_hpf:
    def __init__(self, hs):
        self.hs = hs # hs_system
        self.tracjectory = []

    def init_particles(self, modes, state_mean, state_var, N):
        particles= []
        disturbance = np.random.randn(N, len(state_mean))
        disturbance *= np.sqrt(state_var)
        for i in range(N):
            states = state_mean + disturbance[i]
            ptc = hybrid_particle(self.hs.mode_names,\
                                  self.hs.state_names,\
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
            modes = self.hs.modes(p.mode_values, self.hs.state_values)
            states, output = self.hs.states(modes, self.hs.state_values)
            p.set_hs(modes, states)
            # compute Pobs
            res = (obs - output)**2/self.hs.obs_var
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
