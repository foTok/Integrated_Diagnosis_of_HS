'''
This document implementes some particle filter algorithms.
'''
import os
import torch
import progressbar
import logging
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from math import log
from math import exp
from cnn_gru_diagnoser import ann_step
from utilities import np2tensor
from utilities import smooth
from utilities import window_smooth
from utilities import index
from utilities import dis_sample
from utilities import exp_confidence
from utilities import chi2_confidence
from utilities import normalize
from utilities import resample
from utilities import particle
from utilities import hs_system_wrapper

class ipf:
    def __init__(self, hs, state_sigma, obs_sigma, conf=chi2_confidence):
        self.N = None
        self.obs = None
        self.mode0 = None
        self.state_mu0 = None
        self.state_sigma0 = None
        self.confidence = conf
        self.hsw = hs_system_wrapper(hs, state_sigma, obs_sigma)
        self.mode_detector = None
        self.pf_isolator = None
        self.pf_identifier = []
        self.obs_scale = np.ones(len(self.hsw.obs_sigma))
        self.obs = None
        self.fault_para = np.zeros(len(self.hsw.para_faults()))
        self.tracjectory = []
        self.res = []
        self.mode = None
        self.state = []
        self.para = []
        self.t = 0 # time stamp
        self.output_names = None
        self.para_fault_id = []
        self.latest_sp = 0 # latest switch point
        self.stop_fault_process = False
        self.fault_time = 0

    def set_output_names(self, names):
        self.output_names = names

    def set_scale(self, obs_scale):
        self.obs_scale = obs_scale

    def load_mode_detector(self, file_name):
        if os.path.exists(file_name):
            self.mode_detector = torch.load(file_name, map_location='cpu')
            self.mode_detector.eval()
        else:
            msg = 'warning: model file does not exist, it is not changed.'
            self.log_msg(msg)

    def load_pf_isolator(self, file_name):
        if os.path.exists(file_name):
            isolator = torch.load(file_name, map_location='cpu')
            isolator.eval()
            self.pf_isolator = ann_step(isolator)
        else:
            msg = 'warning: model file does not exist, it is not changed.'
            self.log_msg(msg)

    def load_identifier(self, file_name):
        for name in file_name:
            if os.path.exists(name):
                identifier = torch.load(name, map_location='cpu')
                identifier.eval()
                identifier = ann_step(identifier)
                self.pf_identifier.append(identifier)
            else:
                msg = 'warning: model file does not exist, it is not changed.'
                self.log_msg(msg)

    def estimate_mode(self):
        n_obs = self.obs / self.obs_scale
        t_len, o_len = n_obs.shape
        obs = n_obs.reshape(1, t_len, o_len)
        obs = np2tensor(obs, use_cuda=False)
        mode = self.mode_detector(obs)
        mode = mode.detach().numpy()[0]
        mode = np.argmax(mode, axis=1)
        mode[:50] = self.mode0
        mode = smooth(mode, 50)
        self.mode = mode

    def estimate_fault_paras(self, switch_window=1, verify_window=2, p_thresh=0.05):
        if self.t - 2*verify_window - switch_window < self.latest_sp:
            return
        window_len = int(verify_window / self.hsw.step_len)
        if (np.array(self.para_fault_id[-2*window_len:])==0).any():
            return
        para1 = np.array(self.para[-window_len:])
        para2 = np.array(self.para[-2*window_len:-window_len])
        # add a small number to the first time step to avoid numberic problems.
        para1[0,:] = para1[0,:] + 1e-4
        para2[0,:] = para2[0,:] + 1e-4
        _, p_values = stats.f_oneway(para1, para2)
        where_are_nan = np.isnan(p_values)
        p_values[where_are_nan] = 1
        if (p_values > p_thresh).all():
            para_2w = np.array(self.para[-2*window_len:])
            para = np.mean(para_2w, 0)
            para = np.array([(p if p>0.01 else 0) for p in para])
            para_sigma = np.std(para_2w, 0)*(para!=0)/np.sqrt(2*window_len)
            self.fault_para = para
            self.stop_fault_process = True
            self.fault_time = self.find_fault_time()
            msg = 'A fault occurred at {}s, estimated its magenitude at {}s, fault parameters are mu={}, sigma={}.'\
                  .format(round(self.fault_time, 2), round(self.t, 2), np.round(para, 4), np.round(para_sigma, 4))
            self.log_msg(msg)

    def init_particles(self):
        state_mu = self.state_mu0
        state_sigma = self.state_sigma0
        N = self.N
        particles= []
        disturbance = np.random.randn(N, len(state_mu))*state_sigma
        for i in range(N):
            state = state_mu + disturbance[i]
            ptc = particle(state, weight=1/N)
            particles.append(ptc)
        return particles

    def step_particle(self, ptc, obs, mode_i0, mode):
        fault_para = self.fault_para
        p = ptc.clone()
        # predict state
        state = self.hsw.reset_state(mode_i0, mode, p.state)
        state = self.hsw.state_step(mode, state, fault_para)
        # add process noise
        process_noise = np.random.standard_normal(len(state))*self.hsw.state_sigma
        state += process_noise
        p.set_state(state)
        # compute outputs
        output = self.hsw.output(mode, state, self.output_names)
        # compute Pobs
        res = (obs - output)/self.hsw.obs_sigma # residual square
        Pobs = self.confidence(np.sum(res**2), len(res))
        # weighted res
        res = p.weight*res
        p.set_weigth(p.weight*Pobs)
        return p, res

    def step_diagnoser(self, n_res):
        res = n_res * self.hsw.obs_sigma / self.obs_scale
        self.pf_isolator.step(res)
        for identifier in self.pf_identifier:
            identifier.step(res)

    def step(self, particles, obs, mode):
        '''
        particles: particle list
        '''
        self.t += self.hsw.step_len
        mode_i0 = self.mode0 if not self.state else self.mode[len(self.state)-1]
        self.latest_sp = self.latest_sp if mode_i0==mode else self.t
        particles_ip1 = []
        res = np.zeros(len(self.hsw.obs_sigma))
        for ptc in particles:
            p, r = self.step_particle(ptc, obs, mode_i0, mode)
            particles_ip1.append(p)
            res += r
        normalize(particles_ip1)
        re_particles_ip1 = resample(particles_ip1)
        ave_state = self.ave_state(re_particles_ip1)
        self.tracjectory.append(re_particles_ip1)
        self.state.append(ave_state)
        self.res.append(res)
        self.process_fault(res)

    def detect_mode(self, i):
        '''
        i: the index of time step.
        '''
        mode = self.mode[i]
        return mode

    def detect_para_fault(self):
        pf_dis = self.pf_isolator.latest_out()
        i = np.argmax(pf_dis)
        return i

    def identify_fault_para(self):
        '''
        i: the returned value of detect_para_fault
        '''
        i =  self.detect_para_fault()
        self.para_fault_id.append(i)
        if i==0:
            self.para.append(self.fault_para)
        else:
            i -= 1
            fault_para = np.zeros(len(self.hsw.para_faults()))
            identifier = self.pf_identifier[i]
            f = identifier.latest_out()[0]
            fault_para[i] = f
            self.para.append(fault_para)

    def process_fault(self, res):
        if self.stop_fault_process:
            self.para_fault_id.append(self.para_fault_id[-1])
            self.para.append(self.fault_para)
            return
        self.step_diagnoser(res)
        self.identify_fault_para()
        self.estimate_fault_paras()

    def last_particles(self):
        particles = self.tracjectory[-1] if self.tracjectory else self.init_particles()
        return particles

    def find_fault_time(self, window1=1, window2=4):
        window_len1 = int(window1/self.hsw.step_len)
        window_len2 = int(window2/self.hsw.step_len)
        for i in range(window_len1, len(self.para_fault_id)):
            if (np.array(self.para_fault_id[i-window_len1:i])==0).all() and \
               (np.array(self.para_fault_id[i:i+window_len2])!=0).all():
               return (i+1)*self.hsw.step_len
        return 0


    def track(self, mode, state_mu, state_sigma, obs, N):
        msg = 'Tracking hybrid states...'
        self.log_msg(msg)
        self.mode0, self.state_mu0, self.state_sigma0, self.obs, self.N = mode, state_mu, state_sigma, obs, N
        length = len(obs)
        self.estimate_mode()
        with progressbar.ProgressBar(max_value=length*self.hsw.step_len, redirect_stdout=True) as bar:
            i = 0
            while i < length:
                obs = self.obs[i]
                # discrete mode
                mode = self.detect_mode(i)
                particles = self.last_particles()
                self.step(particles, obs, mode)
                bar.update(float('%.2f'%((i+1)*self.hsw.step_len)))
                i += 1

    def fault_info(self):
        return self.fault_time, self.fault_para

    def ave_state(self, ptcs):
        return sum([p.weight*p.state for p in ptcs])

    def plot_state(self, file_name=None):
        data = np.array(self.state)
        self.hsw.plot_states(data, file_name)

    def plot_mode(self, file_name=None):
        data = np.array(self.mode)
        self.hsw.plot_modes(data, file_name)

    def plot_res(self, file_name=None):
        res = np.array(self.res)
        self.hsw.plot_res(res, file_name)

    def plot_para(self, file_name=None):
        para = np.array(self.para)
        window_smooth(para, 100, 500, 100)
        self.hsw.plot_paras(para, file_name)

    def plot_para_fault(self):
        x = np.arange(len(self.para_fault_id))*self.hsw.step_len
        pf = np.array(self.para_fault_id)
        window_smooth(pf, 100, 500, 100)
        pf = smooth(pf, 50)
        plt.plot(x, pf)
        plt.show()

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
