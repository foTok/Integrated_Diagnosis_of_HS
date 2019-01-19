import numpy as np

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
