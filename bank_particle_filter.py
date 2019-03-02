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
        mode_num = len(self.hsw.mode_names())
        particle = []
        disturbance = np.random.randn(N, len(state_mu))*state_sigma
        for m in range(mode_num):
            p_m = []
            for i in range(N):
                state = state_mu + disturbance[i]
                p = hybrid_particle(m, state, 1/(N*mode_num))
                p_m.append(p)
            particle.append(p_m)
        return particle

