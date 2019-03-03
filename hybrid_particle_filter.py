import progressbar
import logging
import numpy as np
from collections import Counter
from utilities import hs_system_wrapper
from utilities import exp_confidence
from utilities import hybrid_particle
from utilities import normalize
from utilities import resample
from utilities import smooth
from utilities import dis_sample

class hpf: # hybrid particle filter
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

    def set_mode_num(self, mode_num):
        self.mode_num = mode_num

    def init_particle(self):
        state_mu = self.state_mu0
        state_sigma = self.state_sigma0
        N = self.N
        particle = []
        disturbance = np.random.randn(N, len(state_mu))*state_sigma
        for i in range(N):
            state = state_mu + disturbance[i]
            p = hybrid_particle(self.mode0, state, 1/N)
            particle.append(p)
        return particle

    def last_particle(self):
        particle = self.tracjectory[-1] if self.tracjectory else self.init_particle()
        return particle

    def step_particle(self, p, obs):
        mode_i0 = p.mode
        mode_dis = self.hsw.stochastic_mode_step(p.mode, p.state)
        mode = dis_sample(mode_dis)[0]
        p = p.clone()
        # predict state
        state = self.hsw.reset_state(mode_i0, mode, p.state)
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

    def step(self, particle, obs):
        self.t += self.hsw.step_len
        particle_ip1 = []
        for ptc in particle:
            p = self.step_particle(ptc, obs)
            particle_ip1.append(p)
        normalize(particle_ip1)
        re_particle_ip1 = resample(particle_ip1, self.N)
        ave_state = sum([p.weight*p.state for p in re_particle_ip1])
        max_mode = [p.mode for p in re_particle_ip1]
        num_counter = Counter(max_mode)
        max_mode = num_counter.most_common(1)[0][0]
        self.tracjectory.append(re_particle_ip1)
        self.state.append(ave_state)
        self.mode.append(max_mode)

    def track(self, mode, state_mu, state_sigma, obs, N):
            msg = 'Tracking hybrid states...'
            self.log_msg(msg)
            self.mode0, self.state_mu0, self.state_sigma0, self.obs, self.N = mode, state_mu, state_sigma, obs, N
            length = len(obs)
            with progressbar.ProgressBar(max_value=length*self.hsw.step_len, redirect_stdout=True) as bar:
                i = 0
                while i < length:
                    obs = self.obs[i]
                    particle = self.last_particle()
                    self.step(particle, obs)
                    bar.update(float('%.2f'%((i+1)*self.hsw.step_len)))
                    i += 1
            self.mode = smooth(np.array(self.mode), 50)

    def plot_state(self, file_name=None):
        data = np.array(self.state)
        self.hsw.plot_states(data, file_name)

    def plot_mode(self, file_name=None):
        data = np.array(self.mode)
        # data = smooth(data, 50)
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
