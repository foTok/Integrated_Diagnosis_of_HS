import numpy as np
import torch
from torch.distributions.normal import Normal
from scipy.stats import norm
from scipy.stats import chi2

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

def add_noise(data, snr_pro_var=None):
    if snr_pro_var is None:
        return data
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    if isinstance(snr_pro_var, np.ndarray):
        disturb = np.sqrt(snr_pro_var)
    elif snr_pro_var < 1 : # proportion
        mean  = np.mean(np.abs(data), 0)
        var = mean * snr_pro_var
        disturb = np.sqrt(var)
    else: # snr
        ratio = 1/10**(snr_pro_var/20)
        std = np.std(data, 0)
        disturb = std*ratio
    noise = np.random.standard_normal(data.shape) * disturb
    data_with_noise = data + noise
    return data_with_noise

def obtain_var(data, snr_pro):
    if snr_pro < 1 : # proportion
        mean  = np.mean(np.abs(data), 0)
        var = mean * snr_pro
    else: # snr
        ratio = 1/10**(snr_pro/10)
        var = np.var(data, 0)
        var = var*ratio
    return var

def smooth(data, N):
    '''
    smooth discrete data in a 1d np.array by 1 order hold.
    '''
    new_data = data[:]
    if N<=1:
        return new_data
    for i in range(len(data)):
        if not (data[i:i+N]==data[i]).all():
            new_data[i] = new_data[i-1] if i>0 else new_data[i]
    return new_data

def dynamic_smooth(data, N):
    # data: list of np.arrays.
    assert N>=1
    if len(data) < N+2:
        return # do nothing
    last_N2p = data[-N-2]
    last_N1p = data[-N-1] # data at -(N+1)
    last_Ns = np.array(data[-N:])
    res_num = len(last_N1p)
    smooth = np.zeros(res_num)
    for i in range(res_num):
        ns = last_Ns[:, i]
        n1p = last_N1p[i]
        n2p = last_N2p[i]
        smooth[i] = n1p if (ns==n1p).all() else n2p
    data[-N-1] = smooth

def one_mode_cross_entropy(y_head, y, mask=None):
    '''
    args:
        y_head: batch × mode_size
        y: batch × mode_size
    '''
    if mask is not None:
        mask = torch.tensor(mask).float().cuda() if torch.cuda.is_available() else torch.tensor(mask).float()
        indices = (y!=mask).any(1)
        indices2 = (mask!=1)
        y_head = y_head[indices, :]
        y = y[indices, :][:, indices2]
    ce = - y * torch.log(y_head)
    ce = torch.mean(ce, 0)
    ce = torch.sum(ce)
    return ce

def multi_mode_cross_entropy(y_head, y, mask=None):
    '''
    args:
        y_head: the prediceted values
        y: the real values
    '''
    if mask is None:
        mask = [None]*len(y_head)
    ce = torch.tensor(0, dtype=torch.float).cuda() if torch.cuda.is_available() else torch.tensor(0, dtype=torch.float)
    for y1, y0, m in zip(y_head, y, mask):
        y1 = y1.cuda() if torch.cuda.is_available() else y1
        ce += one_mode_cross_entropy(y1, y0, m)
    return ce

def normal_stochastic_loss(mu, sigma, obs, k=1, mask=None):
    if mask is not None:
        mask = torch.tensor(mask).float().cuda() if torch.cuda.is_available() else torch.tensor(mask).float()
        indices = (obs!=mask).any(1)
        mu = mu[indices]
        sigma = sigma[indices]
        obs = obs[indices]
    m = Normal(mu, sigma)
    sample = m.rsample([k])
    # repeat obs
    batch, length = obs.size()
    obs = obs.view(1, batch, length)
    obs = obs.expand(k, batch, length)
    mean_loss = torch.mean((sample-obs)**2, [0, 1])
    sum_loss = torch.sum(mean_loss)
    return sum_loss

def np2tensor(x, use_cuda=True):
    if torch.cuda.is_available() and use_cuda:
        return torch.tensor(x, dtype=torch.float).cuda()
    else:
        return torch.tensor(x, dtype=torch.float)

def window_smooth(data, w1, w2, w3):
    for i in range(w1, len(data)-w2-w3):
        test = (data[i-w1:i]==0).all(0) * (data[i+w2:i+w2+w3]==0).all(0)
        data[i:i+w2] = data[i:i+w2]*(~test)

def Z_test(res, window1, window2, conf=0.99, dynamic=False):
    '''
    |window1|window2|
    if the length of res is less that window1+window2, return 0 directly.
    res: a list or np.array
    '''
    res = np.array(res)
    if len(res) < window1+window2:
        if len(res.shape)==1:
            return 0
        else:
            _, n = res.shape
            return np.array([0]*n)
    # mean and variance of window1
    res1 = res[-(window1+window2):-window2] if dynamic else res[:window1+1]
    mean1, var1 = np.mean(res1, 0), np.var(res1, 0)
    # mean of window2
    res2 = res[-window2:]
    mean2 = np.mean(res2)
    # the var should be modified based on windows
    abs_normalized = abs(mean2 - mean1) / np.sqrt(var1/window2)
    thresh = norm.ppf(1-(1-conf)/2)
    results = (abs_normalized > thresh) + 0
    return results

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
    w = [ptc.weight for ptc in particles]
    sum_w =  sum(w)
    # print('sum_w={}'.format(sum_w), flush=True)
    thresh = 0.250
    w = 0 if sum_w<thresh else sum_w
    state = np.mean([ptc.state for ptc in particles], 0) if sum_w<thresh else None
    for ptc in particles:
        ptc.weight = (ptc.weight / w) if w!=0 else 1/len(particles)
        if state is not None:
            ptc.state = state

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
        ptc.set_weigth(1/N)
        new_partiles.append(ptc)
    return new_partiles

class particle:
    '''
    a particle contains continuous state nad weight
    '''
    def __init__(self, state, weight=1):
        self.state = state
        self.weight = weight

    def set_state(self, state):
        self.state = state

    def set_weigth(self, weight):
        self.weight = weight

    def clone(self):
        pct = particle(self.state, self.weight)
        return pct

class hybrid_particle:
    def __init__(self, mode=None, state=None, weight=None):
        self.mode = mode
        self.state = state
        self.weight = weight

    def set_mode(self, mode):
        self.mode = mode

    def set_state(self, state):
        self.state = state

    def set_weigth(self, weight):
        self.weight = weight

    def clone(self):
        return hybrid_particle(self.mode, self.state, self.weight)

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

    def reset_state(self, mode_i, mode_ip1, state):
        return self.hs.reset_state(mode_i, mode_ip1, state)

    def stochastic_mode_step(self, mode, state):
        return self.hs.stochastic_mode_step(mode, state)

    def state_step(self, mode_ip1, state_i, fault_parameters):
        return self.hs.state_step(mode_ip1, state_i, fault_parameters)

    def output(self, mode, states, output_names=None):
        if output_names is None:
            return self.hs.output(mode, states)
        else:
            return self.hs.output(mode, states, output_names)

    def plot_states(self, states, file_name=None):
        if file_name is None:
            self.hs.plot_states(states)
        else:
            self.hs.plot_states(states, file_name)

    def plot_modes(self, modes, file_name=None):
        if file_name is None:
            self.hs.plot_modes(modes)
        else:
            self.hs.plot_modes(modes, file_name)

    def plot_res(self, res, file_name=None):
        if file_name is None:
            self.hs.plot_res(res)
        else:
            self.hs.plot_res(res, file_name)

    def plot_Z(self, Z, file_name=None):
        if file_name is None:
            self.hs.plot_Z(Z)
        else:
            self.hs.plot_Z(Z, file_name)

    def plot_paras(self, paras, file_name=None):
        if file_name is None:
            self.hs.plot_paras(paras)
        else:
            self.hs.plot_paras(paras, file_name)
