'''
This document implementes some particle filter algorithms.
'''
import os
import sys
rootdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  
sys.path.insert(0, rootdir)
sys.path.insert(0, os.path.join(rootdir, 'ANN'))
import torch
import progressbar
import numpy as np
import matplotlib.pyplot as plt
from math import log
from math import exp
from scipy.stats import chi2
from scipy.stats import norm
from Fault_diagnosis.fault_detector import Z_test
from utilities.utilities import np2tensor
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
    def __init__(self, mode_values, state_values, fault_paras, weight=1):
        self.mode_values = mode_values      # int or list or np.array, may change
        self.state_values = state_values    # list or np.array, may change
        self.fault_paras = fault_paras      # list or np.array, fault parameters, may change
        self.weight = weight                # real, may change

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

    def set_fault_para(self, fault_paras):
        self.fault_paras = fault_paras

    def set_weigth(self, weight):
        self.weight = weight

    def clone(self):
        pct = hybrid_particle(self.mode_values,\
                              self.state_values.copy(),\
                              self.fault_paras.copy(),\
                              self.weight)
        return pct

class hs_system_wrapper:
    '''
    def the interface of hs_system for filter
    '''
    def __init__(self, hs, state_sigma, obs_sigma):
        self.hs = hs
        self.step_len = hs.step_len
        self.state_sigma = state_sigma # np.array
        self.obs_sigma = obs_sigma     # np.array

    def para_faults(self):
        return type(self.hs).f_parameters

    def mode_names(self):
        return type(self.hs).modes

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

    def plot_Z(self, Z):
        self.hs.plot_Z(Z)

    def plot_paras(self, paras):
        self.hs.plot_paras(paras)

class hpf: # hybrid particle filter
    def __init__(self, hsw, conf=chi2_confidence):
        self.N = None
        self.Nmin = None
        self.Nmax = None
        self.obs = None
        self.confidence = conf
        self.hsw = hsw # hs_system_wrapper
        self.identifier = None
        self.state_scale = np.ones(len(self.hsw.state_sigma))
        self.obs_scale = np.ones(len(self.hsw.obs_sigma))
        self.paras_sigma = np.zeros(len(self.hsw.para_faults()))
        self.tracjectory = []
        self.res = []
        self.states = []
        self.modes = []
        self.paras = []
        self.Z = []
        self.t = 0 # time stamp
        self.fd_close_time = None # time of fault detect closed
        self.pf_close_time = None # time of particle fileter closed
        self.fp_open_time = None # time of fault parameter learn open
        self.fd_close_window = 10
        self.pf_close_window = 0
        self.fp_open_window = 8
        self.tmp_fault_paras = None

    def set_scale(self, state_scale, obs_scale):
        self.state_scale = state_scale
        self.obs_scale = obs_scale

    def load_identifier(self, file_name):
        if os.path.exists(file_name):
            self.identifier = torch.load(file_name)
            self.identifier.eval()
        else:
            print('warning: model file does not exist, it is not changed.')

    def fd_is_closed(self):
        r = False
        if self.fd_close_time is not None:
            if self.t < self.fd_close_time + self.fd_close_window:
                r = True
            else :
                self.fd_close_time = None
        return r

    def pf_is_closed(self):
        r = False
        if self.pf_close_time is not None:
            if self.t < self.pf_close_time + self.pf_close_window:
                r = True
            else:
                self.pf_close_time = None
        return r

    def fp_is_open(self):
        r = False
        if self.fp_open_time is not None:
            if self.t < self.fp_open_time + self.fp_open_window:
                r = True
            else:
                self.fp_open_time = None
                self.N = self.Nmin
        return r

    def close_fd(self, t):
        self.fd_close_window = t
        self.fd_close_time = self.t

    def close_pf(self, t):
        self.pf_close_window = t 
        self.pf_close_time = self.t

    def open_fp(self, t):
        self.fp_open_window = t
        self.fp_open_time = self.t
        self.N = self.Nmax
        self.tmp_fault_paras = []

    def collect_fault_paras(self, fault_paras):
        if self.tmp_fault_paras is not None:
            self.tmp_fault_paras.append(fault_paras)

    def estimate_fault_paras(self):
        if (self.tmp_fault_paras is None) or self.fp_is_open():
            return None
        paras = np.array(self.tmp_fault_paras)
        n = (len(paras)*0.25) # the first 1/4 paras are not used because they are unstable.
        paras = paras[n:,:]
        paras = np.mean(paras, 0)
        paras = np.array([(p if p>0.01 else 0) for p in paras])
        self.tmp_fault_paras = None
        print('\nFault paras estimated by PF are {}.'.format(paras), flush=True)
        return paras

    def init_particles(self, modes, state_mean, state_var, N):
        particles= []
        disturbance = np.random.randn(N, len(state_mean))
        disturbance *= np.sqrt(state_var)
        for i in range(N):
            states = state_mean + disturbance[i]
            fault_paras = np.zeros(len(self.hsw.para_faults()))
            ptc = hybrid_particle(modes,\
                                  states,\
                                  fault_paras,\
                                  1/N)
            particles.append(ptc)
        return particles

    def sample_particle_from_ann(self, modes, paras, state_values, para_values, weight):
        modes, has_fault = self.sample_modes(modes)
        states = self.sample_states(state_values)
        paras = self.sample_paras(paras, para_values, has_fault)
        ptc = hybrid_particle(modes,\
                              states,\
                              paras,\
                              weight)
        return ptc

    def sample_modes(self, modes):
        has_fault = False
        mode_names = self.hsw.mode_names()
        new_modes = []
        for m, n in zip(modes, mode_names):
            if has_fault: # single fault assumption
                m = [ m[i] for i in range(len(n)) if not n[i].startswith('s_')]
            sample = dis_sample(m)[0]
            new_modes.append(sample)
            if mode_names[n][sample].startswith('s_'):
                has_fault = True
        new_modes = new_modes[0] if len(new_modes)==1 else np.array(new_modes)
        return new_modes, has_fault

    def sample_states(self, states):
        mu, sigma = states
        rd = np.random.randn(len(mu))
        sample = rd*sigma + mu
        sample = sample * self.state_scale # norm
        return sample

    def sample_paras(self, paras, para_values, has_fault):
        mu, sigma = para_values
        paras = paras if len(paras)==len(mu) else paras[1:]
        fault_paras = np.zeros(len(mu))
        if not has_fault: # no discrete mode fault.
            i = dis_sample(paras)[0]
            rd = np.random.randn()
            fp = rd*sigma[i] + mu[i]
            fp = np.clip(fp, 0, 1) # make sure fp belongs to [0, 1]
            fault_paras[i] = fp
        return fault_paras

    def step_particle(self, ptc, obs, ref_fault_paras):
        p = ptc.clone()
        # one step based on the particle
        modes, states = self.hsw.mode_step(p.mode_values, p.state_values)
        # add noise to the particle
        fault_paras_noise = (p.fault_paras!=0)*np.random.standard_normal(len(p.fault_paras))*self.paras_sigma if self.fp_is_open() else np.zeros(len(p.fault_paras))
        fault_paras_base = p.fault_paras if ref_fault_paras is None else ref_fault_paras
        fault_paras = np.clip(fault_paras_base + fault_paras_noise, 0, 1)
        states = self.hsw.state_step(modes, states, fault_paras)
        # add process noise
        process_noise = np.random.standard_normal(len(states))*self.hsw.state_sigma if not self.pf_is_closed() else np.zeros(len(states))
        states += process_noise
        p.set_hs(modes, states)
        output = self.hsw.output(modes, states)
        # compute Pobs
        res = (obs - output)/self.hsw.obs_sigma # residual square
        Pobs = self.confidence(np.sum(res**2), len(res))
        p.set_weigth(p.weight*Pobs)
        p.set_fault_para(fault_paras)
        return p, res

    def step(self, particles, obs):
        '''
        particles: hybrid_particle list
        '''
        self.t += self.hsw.step_len
        ref_fault_paras = self.estimate_fault_paras()
        particles_ip1 = []
        res = []
        for ptc in particles:
            p, r = self.step_particle(ptc, obs, ref_fault_paras)
            particles_ip1.append(p)
            res.append(r)
        normalize(particles_ip1)
        res = np.sum([p.weight*r for p, r in zip(particles_ip1, res)], 0)
        re_particles_ip1 = resample(particles_ip1, self.N)
        return re_particles_ip1, res

    def detect_fault(self, t1, proportion):
        N = int(t1/self.hsw.step_len)
        if len(self.Z)<N:
            return False
        Z = np.array(self.Z[-N-1:])
        Z = (np.mean(Z, 0)>=proportion)
        r = (True in Z)
        if r:
            print('\nDetect Fault at %.2f s' % self.t)
        return r

    def identify_fault(self, hs0, x):
        # hs0
        n, = hs0.shape
        hs0 = hs0.reshape(1,n)
        # x
        time, obs = x.shape
        x = x.reshape(1, time, obs)
        modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma) = self.identifier((np2tensor(hs0), np2tensor(x)))
        modes = [m.detach().numpy() for m in modes]
        paras = paras.detach().numpy()
        states_mu, states_sigma = states_mu.detach().numpy(), states_sigma.detach().numpy()
        paras_mu, paras_sigma = paras_mu.detach().numpy(), paras_sigma.detach().numpy()
        # reduce the first dimensional
        modes = [m[0,:] for m in modes]
        paras = paras[0,:]
        states_mu, states_sigma = states_mu[0,:], states_sigma[0,:]
        paras_mu, paras_sigma = paras_mu[0,:], paras_sigma[0,:]
        self.paras_sigma = paras_sigma
        return modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma)

    def hs0(self, N):
        '''
        jump back N time steps
        '''
        m0 = self.modes[-N]
        s0 = self.states[-N]
        s0 = s0 / self.state_scale # normal
        if isinstance(m0, int):
            hs0 = np.array([m0] + list(s0))
        else:
            hs0 = np.array(list(m0) + list(s0))
        return hs0

    def fault_process(self, t0, t1, fd, pf, fp, proportion):
        if self.identifier is None:
            return None
        if self.fd_is_closed():
            return None
        has_fault = self.detect_fault(t1, proportion)
        if not has_fault:
            return None
        else:
            self.close_fd(fd)
            self.close_pf(pf)
            self.open_fp(fp)
            particles = []
            N = int((t0 + t1)/self.hsw.step_len)
            # jump back and find the initial states
            hs0 = self.hs0(N+1)
            # obtain the observations
            current = len(self.modes)-1
            x = self.obs[current-N:current, :]
            x = x / self.obs_scale
            # use ann to estimate
            modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma) = self.identify_fault(hs0, x)
            # reset the tracjectories based on estimated values
            # debug
            print('\nANN estimated results: modes={},paras={},state_mu={},state_sigma={},para_mu={},para_sigma={}'\
                  .format(modes, paras, states_mu, states_sigma, paras_mu, paras_sigma), flush=True)
            # resample particles from the estimated values
            for _ in range(self.N):
                ptc = self.sample_particle_from_ann(modes, paras, (states_mu, states_sigma), (paras_mu, paras_sigma), 1/self.N)
                particles.append(ptc)
            return particles

    def last_particles(self, limit, modes, state_mean, state_var, fd, pf, fp, proportion):
        particles = self.fault_process(limit[0], limit[1], fd, pf, fp, proportion)
        if particles is not None:
            return particles
        particles = self.tracjectory[-1] if self.tracjectory else self.init_particles(modes, state_mean, state_var, self.N)
        return particles

    def track(self, modes, state_mean, state_var, observations, limit, fd, pf, fp, proportion, Nmin, Nmax=None):
        print('Tracking hybrid states...')
        self.obs = observations
        self.N = Nmin
        self.Nmin = Nmin
        self.Nmax = 2*Nmin if Nmax is None else Nmax
        progressbar.streams.wrap_stderr()
        with progressbar.ProgressBar(max_value=100) as bar:
            i, obs_len = 0, len(observations)
            while i < obs_len:
                obs = observations[i]
                particles = self.last_particles(limit, modes, state_mean, state_var, fd, pf, fp, proportion)
                particles_ip1, res = self.step(particles, obs)
                ave_states = self.ave_states(particles_ip1)
                ave_paras = self.ave_paras(particles_ip1)
                probable_modes = self.probable_modes(particles_ip1)
                self.states.append(ave_states)
                self.paras.append(ave_paras)
                self.collect_fault_paras(ave_paras)
                self.modes.append(probable_modes)
                self.tracjectory.append(particles_ip1)
                self.res.append(res)
                z = Z_test(self.res, 1000, 10)
                self.Z.append(z)
                bar.update(float('%.2f'%((i+1)*100/obs_len)))
                i += 1

    def ave_states(self, ptcs):
        return sum([p.weight*p.state_values for p in ptcs])

    def ave_paras(self, ptcs):
        return sum([p.weight*p.fault_paras for p in ptcs])

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
        data = np.array(self.states)
        self.hsw.plot_states(data)

    def plot_modes(self, N=50):
        data = np.array(self.modes)
        data = smooth(data, N)
        self.hsw.plot_modes(data)

    def plot_res(self):
        res = np.array(self.res)
        self.hsw.plot_res(res)

    def plot_Z(self, N=50):
        Z = np.array(self.Z)
        Z = smooth(Z, N)
        self.hsw.plot_Z(Z)

    def plot_paras(self):
        paras = np.array(self.paras)
        self.hsw.plot_paras(paras)
