'''
This document implementes some particle filter algorithms.
'''
import threading
import numpy as np
import matplotlib.pyplot as plt
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
        ptc.weight = (ptc.weight / w) if w!=0 else 1/len(particles)

def resample(particles, N):
    def index(s, interval):
        for i, t in zip(range(len(interval)-1), interval[1:]):
            if s < t:
                return i
        return -1

    interval = [0]
    for ptc in particles:
        interval.append(interval[-1]+ptc.weight)
    samples = np.random.uniform(interval[0], interval[-1], N)
    new_partiles = []
    for s in samples:
        i = index(s, interval)
        ptc = particles[i].clone()
        new_partiles.append(ptc)
    normalize(new_partiles)
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
    def __init__(self, hs=None, process_var=None, obs_var=None):
        self.hs = hs
        self.pv = process_var # np.array
        self.ov = obs_var     # np.array
        self.p_std = None if process_var is None else np.sqrt(process_var)# np.array

    def fault_parameters(self, mode, fault_type, fault_magnitude):
        return self.hs.fault_parameters(0, mode, fault_type, 1, fault_magnitude)

    def mode_step(self, mode_i, state_i):
        return self.hs.mode_step(mode_i, state_i)

    def state_step(self, mode_ip1, state_i, fault_parameters):
        return self.hs.state_step(mode_ip1, state_i, fault_parameters)

    def output(self, mode, states):
        return self.hs.output(mode, states)

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
            ptc = hybrid_particle(modes,\
                                  states,\
                                  1/N)
            particles.append(ptc)
        return particles

    def step_particle(self, ptc, particles, obs, lock=None):
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
        res = (obs - output)**2/self.hsw.ov
        Pobs = chi2_confidence(np.sum(res), len(res))
        p.set_weigth(p.weight*Pobs)
        if lock is None:
            particles.append(p)
        else:
            lock.acquire()
            particles.append(p)
            lock.release()

    def step(self, particles, obs):
        '''
        particles: hybrid_particle list
        '''
        particles_ip1 = []
        for ptc in particles:
            self.step_particle(ptc, particles_ip1, obs)
        normalize(particles_ip1)
        re_particles_ip1 = resample(particles_ip1, len(particles_ip1))
        return re_particles_ip1

    def parallel_step(self, particles, obs):
        '''
        particles: hybrid_particle list
        '''
        particles_ip1 = []
        thread = []
        shared_resource_lock = threading.Lock()
        for ptc in particles:
            t = threading.Thread(target=self.step_particle , args=(ptc, particles_ip1, obs, shared_resource_lock))
            t.start()
            thread.append(t)
        for t in thread:
            t.join()
        normalize(particles_ip1)
        re_particles_ip1 = resample(particles_ip1, len(particles_ip1))
        return re_particles_ip1

    def track(self, modes, state_mean, state_var, N, observations, parallel=False):
        step = self.step if not parallel else self.parallel_step
        for obs in observations:
            particles = self.tracjectory[-1] if self.tracjectory else self.init_particles(modes, state_mean, state_var, N)
            particles_ip1 = step(particles, obs)
            self.tracjectory.append(particles_ip1)

    def best_trajectory(self):
        best = []
        for ptcs in self.tracjectory:
            best_ptc = max(ptcs, key=lambda p: p.weight)
            best.append(best_ptc.state_values)
        return np.array(best)

    def average_trajectory(self):
        average = []
        for particles in self.tracjectory:
            state = np.zeros(len(self.hsw.pv))
            for p in particles:
                state += p.weight*p.state_values
            average.append(state)
        return np.array(average)

    def plot(self, index, averge=False):
        if averge:
            data = self.average_trajectory()
        else:
            data = self.best_trajectory()
        plt.plot(data[:, index])
        plt.show()

    def plot_mode(self):
        mode = []
        for ptcs in self.tracjectory:
            best_ptc = max(ptcs, key=lambda p: p.weight)
            mode.append(best_ptc.mode_values)
        plt.plot(np.array(mode))
        plt.show()
