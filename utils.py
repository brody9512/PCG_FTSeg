import os
import random
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import neurokit2 as nk
from neurokit2.signal import *

# Import from Directory Architecture
from config import get_args


args = get_args()

seed_=args.seed

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def set_seed(seed=seed_):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    monai.utils.misc.set_determinism(seed=seed)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.dice = monai.losses.DiceLoss()
   
    def forward(self, inputs, targets):
        dice = self.dice(inputs, targets)
        bce = F.binary_cross_entropy(inputs, targets)
        # print(dice,bce)
        return dice + bce

def eval_metrics(gt1, pr1, a, b, c, d, e, f,toler):

    TP = 0
    TP_= 0

    for i in gt1:
        for j in pr1:
            if abs(i - j) <= toler:
                TP += 1
                break

    FP = len(pr1) - TP

    for q in pr1:
        for p in gt1:
            if abs(p - q) <= toler:
                TP_ += 1
                break
    
    FN = len(gt1) -TP_


    if TP + FN == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)
    
    if TP + FP == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if sensitivity + precision == 0:
        f1_score = 0
    else:
        f1_score = (2 * sensitivity * precision) / (sensitivity + precision)

    a += TP
    b += FN
    c += FP
    d += sensitivity
    e += precision
    f += f1_score

    return a, b, c, d, e, f
    
def zscore(arr,mean=None,std=None):
    if mean!=None or std!=mean:
        return (arr-mean)/(std+1e-8)
    else:
        try:
            return (arr-np.mean(arr))/(np.std(arr)+1e-8)
        except:
            return (arr-torch.mean(arr))/(torch.std(arr)+1e-8)
            
def minmax(tensor):
    try:
        return (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))
    except:
        return (tensor-np.min(tensor))/(np.max(tensor)-np.min(tensor))
    


def augment_neurokit(pcg_signal, sr, p=0.2):
    
    if np.random.rand(1) <= p:
        
        noise_shape = ['gaussian', 'laplace']
        n_noise_shape = np.random.randint(0,2)

        noise_amplitude = np.random.rand(1)*.4 #/ noise_frequency
        powerline_amplitude = np.random.rand(1)*.2 #/ powerline_frequency
        artifacts_amplitude = np.random.rand(1)*1 #/ artifacts_frequency
        
        #samples/second
        noise_frequency = np.random.randint(10,50)
        powerline_frequency = np.random.randint(50,60)
        artifacts_frequency= np.random.randint(2,40)
        
        artifacts_number = 10

        pcg_signal = signal_distort(pcg_signal,
                                    sampling_rate=sr,
                                    noise_shape=noise_shape[n_noise_shape],

                                    #the scale of the random function, relative to the standard deviation of the signal).
                                    noise_amplitude=noise_amplitude,# The amplitude of the noise 
                                    powerline_amplitude=powerline_amplitude,
                                    artifacts_amplitude=artifacts_amplitude,

                                    #samples/second
                                    noise_frequency=noise_frequency,
                                    powerline_frequency=powerline_frequency,
                                    artifacts_frequency=artifacts_frequency,

                                    artifacts_number=artifacts_number,# The number of artifact bursts. The bursts have a random duration between 1 and 10% of the signal duration.
                                    linear_drift=False,#Whether or not to add linear drift to the signal.
                                    random_state=42,#None,#42,
                                    silent=True)
    return pcg_signal

def augment_neurokit2(sig, sr, p=0.3):
    
    if np.random.rand(1) <= p:
        beta = (np.random.rand(1)-.5)*4
        amp = np.random.rand(1)*.01

        noise = nk.signal.signal_noise(duration=len(sig)/sr, sampling_rate=sr, beta=beta) * amp
        aug = augment_neurokit(noise, sr=sr)
        result = np.zeros(len(sig))

        result[:len(aug)] = aug

        filt = np.zeros_like(sig)
        result = result * filt

        filt = scipy.ndimage.gaussian_filter1d(filt,11,order=0,mode='nearest')
        #line chart의 noise를 제거하기 위하여 gaussian filter를 사용하였다.
        
        result = sig + result
        return result

    else:
        return sig