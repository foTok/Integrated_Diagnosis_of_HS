import numpy as np
from utilities import hs_system_wrapper
from utilities import exp_confidence
from utilities import hybrid_particle


class bpf: # bank particle filter
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
        self.res = []
        self.mode = []
        self.state = []
        self.t = 0 # time stamp
        self.fault_time = 0

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

    def step_particle(self, p, obs, mode_i0, mode):
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
        res = p.weight*res
        p.set_weigth(p.weight*Pobs)
        return p, res