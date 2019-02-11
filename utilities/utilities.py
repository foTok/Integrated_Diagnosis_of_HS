import numpy as np
import torch

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

def one_mode_cross_entropy(y_head, y, mask=None):
    '''
    args:
        y_head: batch × mode_size
        y: batch × mode_size
    '''
    if mask is not None:
        mask = torch.tensor(mask).float()
        indices = (y!=mask).any(1)
        indices2 = (mask!=1)
        y_head = y_head[indices]
        y = y[indices][indices2]
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
    if mask is not None:
        mask = [None]*len*y_head
    ce = torch.tensor(0, dtype=torch.float)
    for y1, y0, m in zip(y_head, y, mask):
        ce += one_mode_cross_entropy(y1, y0, m)
    return ce

def normal_stochastic_loss(mu, sigma, obs, k=1, mask=None):
    if mask is not None:
        mask = torch.tensor(mask).float()
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
    return sum_loss, mean_loss.detach().numpy()

def np2tensor(x):
    return torch.tensor(x, dtype=torch.float)
