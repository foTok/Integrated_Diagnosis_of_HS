import progressbar
import logging
import numpy as np
from collections import Counter
from utilities import hs_system_wrapper
from utilities import exp_confidence
from utilities import chi2_confidence
from utilities import hybrid_particle
from utilities import normalize
from utilities import resample
from utilities import smooth

class ipf: # bank particle filter
    def __init__(self, hs, state_sigma, obs_sigma, conf=exp_confidence):
        self.N = None
        self.obs = None
        self.mode0 = None
        self.state_mu0 = None
        self.state_sigma0 = None
        self.confidence = conf
        self.hsw = hs_system_wrapper(hs, state_sigma, obs_sigma)
        self.obs = None
        self.fault_para = np.zeros(len(self.hsw.para_faults()))
        self.tracjectory = []
        self.mode = []
        self.state = []
        self.t = 0 # time stamp
        self.mode_num = 0
        self.mode_particle_dict = {}
        self.mode_dist = []
        self.f = 0.001

    def set_mode_num(self, mode_num):
        self.mode_num = mode_num
        self.mode_dist = np.zeros(mode_num)

    def init_particle(self):
        state_mu = self.state_mu0
        state_sigma = self.state_sigma0
        N = self.N
        disturbance = np.random.randn(N, len(state_mu))*state_sigma
        for m in range(self.mode_num):
            self.mode_particle_dict[m] = []
            for i in range(N):
                state = state_mu + disturbance[i]
                p = hybrid_particle(m, state, 1/N)
                self.mode_particle_dict[m].append(p)

    def step_particle(self, p, obs, mode_i0):
        p = p.clone()
        mode = p.mode
        state = p.state
        # predict state
        state = self.hsw.reset_state(mode_i0, mode, state)
        state = self.hsw.state_step(mode, state, self.fault_para)
        # add process noise
        process_noise = np.random.standard_normal(len(state))*self.hsw.state_sigma
        state += process_noise
        p.set_state(state)
        # compute outputs
        output = self.hsw.output(mode, state)
        # compute Pobs
        res = (obs - output)/self.hsw.obs_sigma # residual square
        Pobs = self.confidence(np.sum(res**2), len(res))
        # weighted res
        p.set_weigth(p.weight*Pobs)
        p.set_mode(mode)
        return p

    def interaction(self):
        # mu
        mu = np.zeros((self.mode_num, self.mode_num))
        for i in range(self.mode_num):
            for j in range(self.mode_num):
                pij = (1 - (self.mode_num-1)*self.f) if (i==j) else self.f
                mu[i, j] = pij*self.mode_dist[i]
        c = np.sum(mu, 0)
        mu = mu / c

        pos = np.isnan(mu)
        mu[pos] = 0

        mode_particle_dict = {}
        for j in range(self.mode_num):
            particles = []
            for i in range(self.mode_num):
                for p in self.mode_particle_dict[j]:
                    _p = p.clone()
                    _p.set_weigth(_p.weight*mu[i,j])
                    particles.append(_p)
            mode_particle_dict[j] = resample(particles, self.N)
        self.mode_particle_dict = mode_particle_dict
        return c

    def step(self, obs):
        self.t += self.hsw.step_len
        mode_i0 = self.mode0 if not self.state else self.mode[len(self.state)-1] # lastest mode
        c = self.interaction()
        particle_ip1 = {}
        for m in range(self.mode_num):
            particle_ip1[m] = []
            particle = self.mode_particle_dict[m]
            for ptc in particle:
                p = self.step_particle(ptc, obs, mode_i0)
                particle_ip1[m].append(p)

        weight = [sum([p.weight for p in particle_ip1[m]])*c[m] for m in particle_ip1]
        weight = [w/sum(weight) for w in weight] if sum(weight)!=0 else self.mode_dist
        w_max = max(weight) # maximal weight
        m_opt = weight.index(w_max) # optimal mode

        ave_state = []
        for m in range(self.mode_num):
            _weight = [p.weight for p in particle_ip1[m]]
            if sum(_weight)!=0:
                _weight = [w/sum(_weight) for w in _weight]
                _state = sum([w*p.state for w, p in zip(_weight, particle_ip1[m])])*weight[m]
                ave_state.append(_state)
        ave_state = self.state[-1] if not ave_state else sum(ave_state)
        
        self.mode_dist = weight
        self.tracjectory.append(particle_ip1[m_opt])
        self.state.append(ave_state)
        self.mode.append(m_opt)
        self.mode_particle_dict = particle_ip1

    def track(self, mode, state_mu, state_sigma, obs, N):
            msg = 'Tracking hybrid states...'
            self.log_msg(msg)
            self.mode0, self.state_mu0, self.state_sigma0, self.obs, self.N = mode, state_mu, state_sigma, obs, N
            self.init_particle()
            self.mode_dist[mode] = 1.0
            length = len(obs)
            with progressbar.ProgressBar(max_value=length*self.hsw.step_len, redirect_stdout=True) as bar:
                i = 0
                while i < length:
                    obs = self.obs[i]
                    self.step(obs)
                    bar.update(float('%.2f'%((i+1)*self.hsw.step_len)))
                    i += 1

    def plot_state(self, file_name=None):
        data = np.array(self.state)
        self.hsw.plot_states(data, file_name)

    def plot_mode(self, file_name=None):
        data = np.array(self.mode)
        data = smooth(data, 50)
        self.hsw.plot_modes(data, file_name)

    def log_msg(self, msg):
        print(msg)
        logging.info(msg)

    def evaluate_modes(self, ref_modes):
        modes = np.array(self.mode)
        if len(modes.shape)==1:
            ref_modes = ref_modes.reshape(-1)
        res = np.abs(modes - ref_modes)
        wrong_num = np.sum(res!=0, 0)
        accuracy = np.mean(res==0, 0)
        msg = 'Wrong mode number = {}, accuracy = {}.'.format(wrong_num, np.round(accuracy, 4))
        self.log_msg(msg)
        return wrong_num, accuracy

    def evaluate_states(self, ref_states):
        est_states = np.array(self.state)
        # abs error 
        res = est_states - ref_states
        mu = np.mean(res, 0)
        sigma = np.std(res, 0)
        msg = 'States error mu = {}, sigma = {}.'.format(np.round(mu, 4), np.round(sigma, 4))
        self.log_msg(msg)
        # relative error
        # normalize factor
        ref_mu = np.mean(ref_states, 0)
        ref_sigma = np.std(ref_states, 0)
        # normalize ref states
        n_ref_states = (ref_states - ref_mu)/ref_sigma
        # normalize estimated states
        n_est_states = (est_states - ref_mu)/ref_sigma
        n_res = n_est_states - n_ref_states
        n_mu = np.mean(n_res, 0)
        n_sigma = np.std(n_res, 0)
        msg = 'States error n_mu = {}, n_sigma = {}.'.format(np.round(n_mu, 4), np.round(n_sigma, 4))
        self.log_msg(msg)
        return mu, sigma, n_mu, n_sigma
