from scipy.signal import resample_poly
import numpy as np
from scipy.signal import butter, lfilter, filtfilt
import glob
import librosa as lb
import pandas as pd
import natsort

wav_path_ = natsort.natsorted(glob('/home/brody9512/workspace/changhyun/PCG_infer/vital_ch/PhysioNet2022_DB/*.wav'))
seg_path_ = natsort.natsorted(glob('/home/brody9512/workspace/changhyun/PCG_infer/vital_ch/PhysioNet2022_Modified_TSV/*.tsv'))
wav_path_[0].split('/')[-1].split('.')[0]

##(js)##
# Normalize function
def normalize_signal(signal):
    return (signal - np.mean(signal)) / np.max(np.abs(signal))

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 이론 :
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut=None, highcut=None, fs=None, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, cutoff, fs, order):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# Step 3: Create functions for resampling and annotation adjustments
def resample_audio(wav_path, seg_path,resampling,low=None,high=None,lowpass=None,
                   downsampling=False,bandpass=False,up_=None,down_=None):
    wav, orig_sr = lb.load(wav_path, sr=None)
    #print(f"기존의 샘플링 레이트: {orig_sr} Hz")
    orig_samples = len(wav)

    if downsampling:
        # Downsample to 1000 Hz using polyphase antialiasing filter
        wav = resample_poly(wav, up=up_, down=down_)  # original sr was 4000
        
        new_sr = (up_ / down_) * orig_sr
        #print(f"새로운 샘플링 레이트: {new_sr} Hz")

    wav = normalize_signal(wav)

    if bandpass:
        # Apply Butterworth bandpass filter
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        # Apply Butterworth lowpass filter
        wav = butter_lowpass_filter(wav, fs=resampling, cutoff=lowpass, order=5)


    df = pd.read_csv(seg_path,sep='\t',header=None)
    seg = np.zeros_like(wav)
    #print('seg:',seg)#[0. 0. 0. ... 0. 0. 0.]   

    for idx in range(len(df)):
        seg[int(df[0][idx]*resampling):int(df[1][idx]*resampling)] = df[2][idx]
    #np.where(seg!=0): (array([  484,   485,   486, ..., 19624, 19625, 19626]),)
    zeros = np.where(seg!=0)[0]
    
    # Check if 'zeros' is empty
    if zeros.size == 0:
        print(f"No non-zero segment found for file {wav_path}")
        return None, None  # return None values

    wav_new = wav[zeros[0]:zeros[-1]]
    seg_new = seg[zeros[0]:zeros[-1]]

    return wav_new, seg_new
#######
##(js)##
import tqdm.notebook as tqdm

def process_data_resample(low_, high_, sampling_, feature_length):
    up = sampling_
    down = 4000
    idx = 0
    count = 0
    problem_count = 0
    total_resample = []

    #for idx in range(len(seg_path_)):
    for idx in tqdm.trange(len(seg_path_)):
        wav, seg = resample_audio(wav_path_[idx], seg_path_[idx], sampling_, bandpass=True, downsampling=True, low=low_, high=high_, up_=up, down_=down)

        try:
            if 0 in seg:
                count += 1
            else:
                total_resample.append({'wav': wav, 'seg': seg, 'fname': wav_path_[idx].split('/')[-1]})
            # 문제 없는 것은 total_resample에 추가, 문제가 있으면 problem_count에 추가
        except:
            print(f"Exception occurred for file {wav_path_[idx]}")
            problem_count += 1
            pass

    print('total_resample:', len(total_resample))
    print('count:', count)
    print('problem_count:', problem_count,'\n')

    total_new = []
    for t in total_resample:
        if len(t['wav']) >= feature_length:
            total_new.append(t)

    print('total_new:', len(total_new))
    np.save(f'./PhysioNet2022_{sampling_}Hz_{low_}_{high_}_fe_{feature_length}.npy', total_new)

# Define the parameter combinations
resample_params = [
    {'low_': 20, 'high_': 250},
    {'low_': 20, 'high_': 400},
    {'low_': 15, 'high_': 250},
    {'low_': 15, 'high_': 400},
]

# Values of num
num_values = [(1000,6144),(1000,6144),(1000,6144),(1000,6144),#(1000,5120),(1000,3072),(1000,2048),#(5,2048)
             ]
# Iterate over num values
for (num,feature_len) in num_values:
    sampling_ = num
    feature_length = feature_len #round(409.6 * num)
    print(f'sampling_: {sampling_} ,feature_length: {feature_length} ')
    # Process data for each parameter combination
    for params in resample_params:
        process_data_resample(**params, sampling_=sampling_, feature_length=feature_length)
