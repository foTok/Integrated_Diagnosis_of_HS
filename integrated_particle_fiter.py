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
from utilities import Z_test

class ipf:
    def __init__(self, hs, state_sigma, obs_sigma, conf=chi2_confidence):
        self.N = None
        self.Nmin = None
        self.Nmax = None
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
        self.fault_time = -float('inf')
        self.Z = []
        self.obs_conf = 0.05
        self.filter_mode = 'ann' # {'ann', 'Z', 'pf'}
                                 # ann: default mode, all diagnosis processes are accomplished by ann.
                                 # Z: detect mode by ann but detect fault by Z-test. No fault identification after that.
                                 # pf: use ann to detect mode and fault, but employ pf to identify fault.
        # Used by particle filter based parameter estimation.
        self.fault_para_sigma = np.ones(len(self.hsw.para_faults()))*0.01
        self.fault_para_flag = np.zeros(len(self.hsw.para_faults()))
        self.last_likelihood = 1
        self.particle_para_estimation = False
        self.detect_time = -float('inf')

    def set_filter_mode(self, mode):
        self.filter_mode = mode

    def set_obs_conf(self, conf):
        self.obs_conf = conf

    def set_fault_para_sigma(self, sigma):
        self.fault_para_sigma = sigma

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
        mode[:100] = self.mode0
        mode = smooth(mode, 50)
        self.mode = mode

    def estimate_fault_paras(self, switch_window=1, verify_window=2, p_thresh=0.05):
        if self.t - 2*verify_window - switch_window < self.latest_sp:
            return
        window_len = int(verify_window / self.hsw.step_len)
        fault_rate = np.sum(np.array(self.para_fault_id[-2*window_len:])!=0)/(2*window_len)
        if fault_rate < 0.95:
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
            para = np.array([(p if p>0.01 else 0.0) for p in para])
            para_sigma = np.std(para_2w, 0)*(para!=0)/np.sqrt(2*window_len)
            self.fault_para = para
            self.stop_fault_process = True
            self.fault_time = self.find_fault_time()
            msg = 'A fault occurred at {}s, estimated its magnitude at {}s, fault parameters are mu={}, sigma={}.'\
                  .format(round(self.fault_time, 2), round(self.t, 2), np.round(para, 4), np.round(para_sigma, 4))
            self.log_msg(msg)

    def pf_estimate_fault_paras(self, switch_window=1, verify_window=2, p_thresh=0.05):
        if self.t - 2*verify_window - switch_window < self.latest_sp:
            return
        if self.t - self.detect_time < 2*verify_window:
            return
        if (self.fault_para_flag==0).all():
            return
        if self.last_likelihood < 0.95:
            return
        window_len = int(verify_window / self.hsw.step_len)
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
            para = np.array([(p if p>0.01 else 0.0) for p in para])
            para_sigma = np.std(para_2w, 0)*(para!=0)/np.sqrt(2*window_len)
            self.fault_para = para
            self.stop_fault_process = True
            self.fault_para_flag[:] = 0.0
            self.N = self.Nmin
            self.particle_para_estimation = False
            msg = 'A fault occurred at {}s, estimated its magnitude at {}s, fault parameters are mu={}, sigma={}.'\
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
        p = ptc.clone()
        # fault para
        fault_para = self.fault_para if (self.fault_para_flag==0).all() else p.para
        # add para noise
        para_noise = np.random.standard_normal(len(fault_para))*self.fault_para_sigma*self.fault_para_flag
        fault_para += para_noise
        fault_para = np.clip(fault_para, 0, 1)
        p.set_para(fault_para) # set para
        # predict state
        state = self.hsw.reset_state(mode_i0, mode, p.state)
        state = self.hsw.state_step(mode, state, fault_para)
        # add process noise
        process_noise = np.random.standard_normal(len(state))*self.hsw.state_sigma
        state += process_noise
        p.set_state(state) # set state
        # compute outputs
        output = self.hsw.output(mode, state, self.output_names)
        # compute Pobs
        res = (obs - output)/self.hsw.obs_sigma # residual square
        Pobs = self.confidence(np.sum(res**2), len(res))
        # weighted res
        res = p.weight*res
        p.set_weigth(p.weight*Pobs) # set weight
        return p, res

    def step_isolator(self, n_res):
        res = n_res * self.hsw.obs_sigma / self.obs_scale
        self.pf_isolator.step(res)

    def step_identifier(self, n_res):
        res = n_res * self.hsw.obs_sigma / self.obs_scale
        for identifier in self.pf_identifier:
            identifier.step(res)

    def step(self, particles, obs, mode):
        '''
        particles: particle list
        '''
        self.t += self.hsw.step_len
        obs_conf = self.obs_conf if (self.fault_para_flag==0).all() else 0.0
        mode_i0 = self.mode0 if not self.state else self.mode[len(self.state)-1]
        self.latest_sp = self.latest_sp if mode_i0==mode else self.t
        particles_ip1 = []
        res = np.zeros(len(self.hsw.obs_sigma))
        for ptc in particles:
            p, r = self.step_particle(ptc, obs, mode_i0, mode)
            particles_ip1.append(p)
            res += r
        self.last_likelihood = sum([ptc.weight for ptc in particles_ip1])
        normalize(particles_ip1, obs_conf)
        re_particles_ip1 = resample(particles_ip1, self.N)
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

    def pf_identify_fault_para(self, window=2.0):
        i =  self.detect_para_fault()
        self.para_fault_id.append(i)
        window_len = int(window/self.hsw.step_len)
        if i!=0 and (np.array(self.para_fault_id[-window_len:])!=0).all() and \
           abs(self.t-self.latest_sp)>3 and not self.particle_para_estimation:
            self.particle_para_estimation = True
            i -= 1
            self.N = self.Nmax
            particles = self.tracjectory[-1]
            self.detect_time = self.t # detect time
            self.fault_time = self.t - window
            self.fault_para_flag[i] = 1
            particles = resample(particles, self.Nmax)
            for p in particles:
                magnitude = np.random.uniform(0.05, 0.5)
                init_para = np.zeros(len(self.hsw.para_faults()))
                init_para[i] = magnitude
                p.set_para(init_para)
            self.tracjectory[-1] = particles

        if not self.particle_para_estimation:
            fault_para = self.fault_para
        else:
            particles = self.tracjectory[-1]
            fault_para = np.sum([p.weight*p.para for p in particles], 0)
        self.para.append(fault_para)

    def process_fault(self, res):
        if self.stop_fault_process:
            self.para_fault_id.append(self.para_fault_id[-1])
            self.para.append(self.fault_para)
            return
        if self.filter_mode=='ann':
            self.step_isolator(res)
            self.step_identifier(res)
            self.identify_fault_para()
            self.estimate_fault_paras()
        elif self.filter_mode=='Z':
            z = Z_test(res=self.res, window1=1000, window2=100)
            z = z if abs(self.t - self.latest_sp)>3 else (np.zeros(len(z)) if not isinstance(z, int) else 0)
            self.Z.append(z)
            self.check_Z()
        elif self.filter_mode=='pf':
            self.step_isolator(res)
            self.pf_identify_fault_para()
            self.pf_estimate_fault_paras()
        else:
            raise RuntimeError('unknown filter mode.')

    def last_particles(self):
        particles = self.tracjectory[-1] if self.tracjectory else self.init_particles()
        return particles

    def find_fault_time(self, window1=1, window2=4):
        window_len1 = int(window1/self.hsw.step_len)
        window_len2 = int(window2/self.hsw.step_len)
        para_fault_id = np.array(self.para_fault_id)
        para_fault_id = smooth(para_fault_id, 100)
        for i in range(window_len1, len(para_fault_id)):
            if (np.array(para_fault_id[i-window_len1:i])==0).all() and \
               (np.array(para_fault_id[i:i+window_len2])!=0).all():
               return (i+1)*self.hsw.step_len
        return 0

    def check_Z(self, window=2):
        if self.fault_time>0:
            return
        window_len = int(window/self.hsw.step_len)
        if len(self.Z)<=window_len:
            return
        Z = np.array(self.Z[-window_len:])
        Z = np.mean(Z!=0, 0)
        if (Z>0.95).any():
            self.fault_time = self.t - window
            msg = 'A fault occurred at {}s.'.format(round(self.fault_time, 2))
            self.log_msg(msg)

    def track(self, mode, state_mu, state_sigma, obs, N, Nmax):
        msg = 'Tracking hybrid states...'
        self.log_msg(msg)
        self.mode0, self.state_mu0, self.state_sigma0, self.obs, self.N = mode, state_mu, state_sigma, obs, N
        self.Nmin, self.Nmax = N, Nmax
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
        if not self.para:
            return
        para = np.array(self.para)
        window_smooth(para, 100, 500, 100)
        self.hsw.plot_paras(para, file_name)

    
    def plot_Z(self, file_name=None):
        if not self.Z:
            return
        Z = np.array(self.Z)
        self.hsw.plot_Z(Z, file_name)

    def plot_para_fault(self):
        if not self.para_fault_id:
            return
        x = np.arange(len(self.para_fault_id))*self.hsw.step_len
        pf = np.array(self.para_fault_id)
        # window_smooth(pf, 100, 500, 100)
        # pf = smooth(pf, 50)
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
