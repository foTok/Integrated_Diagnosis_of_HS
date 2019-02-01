'''
This document implementes some particle filter algorithms.
'''
import os
import sys
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0,parentdir)
import numpy as np
import progressbar
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy.stats import norm
from Fault_diagnosis.fault_detector import Z_test
from utilities.utilities import smooth

def chi2_confidence(x, df):
    '''
    return the confidence
    x is the value which is normalized.
    df is the freedom
    '''
    return 1 - chi2.cdf(x, df)

def exp_confidence(x, df=None):
    '''
    exp(-0.5*r^2), x=r^2
    df: no significance, keep a consistent interface.
    '''
    return np.exp(-0.5*x)

def normalize(particles):
    w = 0
    for ptc in particles:
        w += ptc.weight
    for ptc in particles:
        ptc.weight = (ptc.weight / w) if w!=0 else 1/len(particles)

def index(s, interval):
    for i, t in zip(range(len(interval)-1), interval[1:]):
        if s < t:
            return i
    return -1

def dis_sample(dis, N=1): # discrete sample
    interval = [0]
    for p in dis:
        interval.append(interval[-1]+p)
    rand_num = np.random.uniform(interval[0], interval[-1], N)
    samples = []
    for r in rand_num:
        i = index(r, interval)
        samples.append(i)
    return samples

def resample(particles, N=None):
    N = len(particles) if N is None else N
    interval = [0]
    for ptc in particles:
        interval.append(interval[-1]+ptc.weight)
    samples = np.random.uniform(interval[0], interval[-1], N)
    new_partiles = []
    for s in samples:
        i = index(s, interval)
        ptc = particles[i].clone()
        ptc.weight = 1/N
        new_partiles.append(ptc)
    return new_partiles

class hybrid_particle:
    '''
    a hybrid particle contains both discrete modes and continuous states
    '''
    def __init__(self, mode_values=None, state_values=None, weight=1):
        self.mode_values = mode_values      # int, may change
        self.state_values = state_values    # list or np.array, may change
        self.weight = weight                # real, may change
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
        self.weight = weight

    def set_fault(self, fault_type, fault_magnitude=None):
        self.fault_type = fault_type
        self.fault_magnitude = fault_magnitude

    def clone(self):
        pct = hybrid_particle(self.mode_values,\
                              self.state_values.copy(),\
                              self.weight)
        return pct

class hs_system_wrapper:
    '''
    def the interface of hs_system for filter
    '''
    def __init__(self, hs, process_var, obs_var):
        self.hs = hs
        self.step_len = hs.step_len
        self.pv = process_var # np.array
        self.ov = obs_var     # np.array

    def fault_parameters(self, mode, fault_type, fault_magnitude):
        return self.hs.fault_parameters(0, mode, fault_type, 1, fault_magnitude)

    def close2switch(self, modes, states):
        return self.hs.close2switch(modes, states)

    def mode_step(self, mode_i, state_i):
        return self.hs.mode_step(mode_i, state_i)

    def state_step(self, mode_ip1, state_i, fault_parameters):
        return self.hs.state_step(mode_ip1, state_i, fault_parameters)

    def output(self, mode, states):
        return self.hs.output(mode, states)

    def plot_states(self, states):
        self.hs.plot_states(states)

    def plot_modes(self, modes):
        self.hs.plot_modes(modes)

    def plot_res(self, res):
        self.hs.plot_res(res)

    def plot_Z(self, res):
        self.hs.plot_Z(res)

class hpf: # hybrid particle filter
    def __init__(self, hsw, conf=chi2_confidence):
        self.N = None
        self.Nmin = None
        self.Nmax = None
        self.confidence = conf
        self.hsw = hsw # hs_system_wrapper
        self.tracjectory = []
        self.res = []
        self.states = []
        self.modes = []
        self.Z = []

    def init_particles(self, modes, state_mean, state_var, N):
        particles= []
        disturbance = np.random.randn(N, len(state_mean))
        disturbance *= np.sqrt(state_var)
        for i in range(N):
            states = state_mean + disturbance[i]
            ptc = hybrid_particle(modes,\
                                  states,\
                                  1/N)
            particles.append(ptc)
        return particles

    def step_particle(self, ptc, obs):
        p = ptc.clone()
        # one step based on the partical
        modes, para_fault = self.hsw.fault_parameters(p.mode_values, p.fault_type, p.fault_magnitude)
        modes, states = self.hsw.mode_step(modes, p.state_values)
        states = self.hsw.state_step(modes, states, para_fault)
        # add process noise
        process_noise = np.random.standard_normal(len(states))*np.sqrt(self.hsw.pv)
        states += process_noise
        p.set_hs(modes, states)
        output = self.hsw.output(modes, states)
        # compute Pobs
        res = (obs - output)/np.sqrt(self.hsw.ov) # residual square
        Pobs = self.confidence(np.sum(res**2), len(res))
        p.set_weigth(p.weight*Pobs)
        return p, res

    def step(self, particles, obs):
        '''
        particles: hybrid_particle list
        '''
        particles_ip1 = []
        res = []
        for ptc in particles:
            p, r = self.step_particle(ptc, obs)
            particles_ip1.append(p)
            res.append(r)
        normalize(particles_ip1)
        res = np.sum([p.weight*r for p, r in zip(particles_ip1, res)], 0)
        re_particles_ip1 = resample(particles_ip1, self.N)
        return re_particles_ip1, res

    def fault_detect(self, N=50):
        if len(self.Z)<N:
            return False
        Z = np.array(self.Z)[-1-N:-1, :]
        Z = [(z==1).all() for z in Z]
        return sum(Z)>=1

    def fault_identifier(self):
        pass

    def track(self, modes, state_mean, state_var, observations, Nmin, Nmax=None):
        self.N = Nmin
        self.Nmin = Nmin
        self.Nmax = 2*Nmin if Nmax is None else Nmax
        progressbar.streams.wrap_stderr()
        with progressbar.ProgressBar(max_value=100) as bar:
            i, obs_len = 0, len(observations)
            while i < obs_len:
                obs = observations[i]
                particles = self.tracjectory[-1] if self.tracjectory else self.init_particles(modes, state_mean, state_var, self.N)
                particles_ip1, res = self.step(particles, obs)
                ave_states = self.ave_states(particles_ip1)
                probable_modes = self.probable_modes(particles_ip1)
                self.states.append(ave_states)
                self.modes.append(probable_modes)
                self.tracjectory.append(particles_ip1)
                self.res.append(res)
                z = Z_test(self.res, 1000, 10)
                self.Z.append(z)
                bar.update(float('%.2f'%((i+1)*100/obs_len)))
                i += 1
            self.states = np.array(self.states)
            self.modes = np.array(self.modes)
            bar.update(100)
            progressbar.streams.flush()

    def ave_states(self, ptcs):
        return sum([p.weight*p.state_values for p in ptcs])

    def probable_modes(self, ptcs):
        mode_dict = {}
        for p in ptcs:
            tuple_p = tuple(p.mode_values) if not isinstance(p.mode_values, int) else p.mode_values
            if tuple_p not in mode_dict:
                mode_dict[tuple_p] = p.weight
            else:
                mode_dict[tuple_p] += p.weight
        probable_modes = max(mode_dict, key=lambda p: mode_dict[p])
        return probable_modes

    def plot_states(self):
        self.hsw.plot_states(self.states)

    def plot_modes(self, N=50):
        modes = smooth(self.modes, N)
        self.hsw.plot_modes(modes)

    def plot_res(self):
        res = np.array(self.res)
        self.hsw.plot_res(res)

    def plot_Z(self, N=50):
        Z = np.array(self.Z)
        Z = smooth(Z, N)
        self.hsw.plot_Z(Z)
