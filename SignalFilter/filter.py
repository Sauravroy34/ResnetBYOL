import numpy as np
import pywt
from scipy.signal import butter, filtfilt, iirnotch, medfilt, find_peaks








def _remove_baseline_wander(ecg_signal, wavelet='sym8'):

    desired  = int(np.ceil(np.log2(100/0.5)))
    max_level = pywt.dwt_max_level(len(ecg_signal),pywt.Wavelet(wavelet).dec_len)
    level =  min(desired,max_level)
    
    coeffs = pywt.wavedec(ecg_signal, wavelet, level=level)
    

    
    coeffs_filt = [coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]]
    
    baseline = pywt.waverec(coeffs_filt, wavelet)
    
    baseline = baseline[:len(ecg_signal)]
    
    return ecg_signal - baseline


def _denoise(data):
    # wavelet transform
    coeffs = pywt.wavedec(data=data, wavelet='db5', level=9)
    cA9, cD9, cD8, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs

    # Threshold denoising
    threshold = (np.median(np.abs(cD1)) / 0.6745) * (np.sqrt(2 * np.log(len(cD1))))
    cD1.fill(0)
    cD2.fill(0)
    for i in range(1, len(coeffs) - 2):
        coeffs[i] = pywt.threshold(coeffs[i], threshold)

    # Inverse wavelet transform to obtain the denoised signal
    rdata = pywt.waverec(coeffs=coeffs, wavelet='db5')
    return rdata


def remove_noise_and_wander(data):
    baseline_wander = _remove_baseline_wander(data)
    cleaned = _denoise(baseline_wander)

    return cleaned