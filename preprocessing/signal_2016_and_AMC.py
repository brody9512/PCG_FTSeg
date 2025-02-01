##(js)##
from scipy.signal import butter, lfilter, filtfilt
import librosa as lb
import numpy as np
import os
import glob
import scipy


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



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 이론 :
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a



def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# y = filtfilt(b,a,x)는 입력 데이터 x를 순방향과 역방향 모두로 처리하여 영위상 디지털 필터링을 수행합니다.



#cutoff는 이 주파수 이상에서 신호를 감쇠시키고자 하는 값입니다.

#Nyquist 주파수는 fs의 절반인 샘플링 비율의 절반을 계산하고 nyq에 저장합니다. 
# 이것은 제공된 샘플링 비율로 정확하게 표현할 수 있는 최대 주파수

#fs는 데이터의 샘플링 비율입니다.
def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # Nyquist Frequency 

    #cutoff 주파수는 Nyquist 주파수로 정규화되어 butter 함수에 의해 필요로하는 0과 1 사이에 있도록 보장합니다. 
    # 이 normal_cutoff 값은 디지털 필터의 중요한 주파수 명세 
    
    # normal_cutoff는 1보다 낮아야 한다
    normal_cutoff = cutoff / nyq
    # the normal_cutoff should be between 0 and 1
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #butter:버터워스 필터는 통과 대역에서 가능한 한 평탄한 주파수 응답을 갖도록 설계된 일종의 신호 처리 필터입니다.
    y = filtfilt(b, a, data)
    return y


def load_data_2016(wav_path,seg_path,resampling,low=None,high=None,lowpass=None,
                   downsampling=False,bandpass=False,up_=None,down_=None):
    # Load wav file
    wav, orig_sr = lb.load(wav_path, sr=None)

    if downsampling:
        # Downsample to 1000 Hz using polyphase antialiasing filter
        wav = resample_poly(wav, up=resampling, down=orig_sr)  # original sr was 2000

    if bandpass:
        # Apply Butterworth bandpass filter
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        # Apply Butterworth lowpass filter
        wav = butter_lowpass_filter(wav, fs=resampling, cutoff=lowpass, order=5)

    df = scipy.io.loadmat(seg_path)['state_ans']
    seg = np.zeros_like(wav)

    for idx in range(len(df)-1):
        start = int(df[idx,0][0,0]*resampling/orig_sr)  # convert seconds to sample indices
        end = int(df[idx+1,0][0,0]*resampling/orig_sr)  # convert seconds to sample indices
        cls = df[idx,1][0,0][0]

        if cls == 'diastole':
            cls = 1
        elif cls == 'S1':
            cls = 2
        elif cls == 'systole':
            cls = 3
        elif cls == 'S2':
            cls = 4

        seg[start:end] = int(cls)

    zeros = np.where(seg!=0)[0]
    seg = seg[zeros[0]:zeros[-1]]
    wav = wav[zeros[0]:zeros[-1]]

    return wav, seg

##(js)##
import tqdm.notebook as tqdm

def process_data(low, high, sampling, feature_length):
    total_data = []

    for idx in tqdm.trange(len(a_01)):
        wav, seg = load_data_2016(a_01[idx], b_01[idx], sampling, low=low, high=high, bandpass=True, downsampling=True, up_=None, down_=None)
        total_data.append({'wav': wav, 'seg': seg, 'fname': a_01[idx].split('/')[-1]})

    total_feature_length_data = []
    for t in total_data:
        if len(t['wav']) >= feature_length:
            total_feature_length_data.append(t)

    print(len(total_feature_length_data))
    np.save(f'/home/brody9512/workspace/changhyun/PCG_infer/PhysioNet2016_{sampling}Hz_{low}_{high}_fe_{feature_length}.npy', total_feature_length_data)

# Define the parameter combinations
parameter_combinations = [
    # (20, 200, 500, round(409.6*5)),
    # (20, 200, 900, round(409.6*9)),
    # (20, 200, 800, round(409.6*8)),
    # (20, 200, 700, round(409.6*7)),
    # (20, 200, 600, round(409.6*6)),

    (20, 250, 1000, 6144),
    (20, 400, 1000, 6144),
    (15, 250, 1000, 6144),
    (15, 400, 1000, 6144),
    # (20, 200, 1000, 5120),
    # (20, 200, 1000, 3072),
    # (20, 200, 1000, 2048),
    # (20, 200, 500, 2048),
]


# Process data for each parameter combination
for low_, high_, sampling_, feature_length_ in parameter_combinations:
    process_data(low_, high_, sampling_, feature_length_)

##(js)##
from scipy.signal import butter, lfilter, filtfilt
from scipy.signal import resample_poly

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



def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs  # 나이퀴스트 이론 :
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a



def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# y = filtfilt(b,a,x)는 입력 데이터 x를 순방향과 역방향 모두로 처리하여 영위상 디지털 필터링을 수행합니다.



#cutoff는 이 주파수 이상에서 신호를 감쇠시키고자 하는 값입니다.

#Nyquist 주파수는 fs의 절반인 샘플링 비율의 절반을 계산하고 nyq에 저장합니다. 
# 이것은 제공된 샘플링 비율로 정확하게 표현할 수 있는 최대 주파수

#fs는 데이터의 샘플링 비율입니다.
def butter_lowpass_filter(data, fs, cutoff, order):
    nyq = 0.5 * fs  # Nyquist Frequency 

    #cutoff 주파수는 Nyquist 주파수로 정규화되어 butter 함수에 의해 필요로하는 0과 1 사이에 있도록 보장합니다. 
    # 이 normal_cutoff 값은 디지털 필터의 중요한 주파수 명세 
    
    # normal_cutoff는 1보다 낮아야 한다
    normal_cutoff = cutoff / nyq
    # the normal_cutoff should be between 0 and 1
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    #butter:버터워스 필터는 통과 대역에서 가능한 한 평탄한 주파수 응답을 갖도록 설계된 일종의 신호 처리 필터입니다.
    y = filtfilt(b, a, data)
    return y


def load_data_2016_amc(sig_path,resampling, orig_sr=4000,low=None,high=None,lowpass=None,
                   downsampling=False,bandpass=False,up_=None,down_=None):
    
        
    data = np.load(sig_path)


    sig = data['sig']
    # Make sure your signal is of dtype float32
    sig = np.float32(sig / np.max(np.abs(sig)))
    
    # # Load sig file
    # sig, orig_sr = lb.load(sig_path, sr=None)

    if downsampling:
        # Downsample to 1000 Hz using polyphase antialiasing filter
        sig = resample_poly(sig, up=resampling, down=orig_sr)  # original sr was 2000

    if bandpass:
        # Apply Butterworth bandpass filter
        sig = butter_bandpass_filter(sig, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        # Apply Butterworth lowpass filter
        sig = butter_lowpass_filter(sig, fs=resampling, cutoff=lowpass, order=5)


    s1 = data['s1']
    s2 = data['s2']
    sys = []
    dia = []

    # Loop through s1 to find the corresponding s2 value that matches the criteria for sys_idx, with rounding
    for i in range(len(s1)):
        # Find the s2 value that is closest but not lower than the current s1 value
        s2_candidates = s2[s2 >= s1[i]]
        if len(s2_candidates) > 0:
            s2_closest = s2_candidates[0]
            sys_idx_rounded = round((s1[i] + s2_closest) / 2)
            sys.append(sys_idx_rounded)

    # Loop through s2 to find the corresponding s1 value that matches the criteria for dia_idx, with rounding
    for i in range(len(s2)):
        # Find the s1 value that is closest but not lower than the current s2 value
        s1_candidates = s1[s1 > s2[i]]
        if len(s1_candidates) > 0:
            s1_closest = s1_candidates[0]
            dia_idx_rounded = round((s2[i] + s1_closest) / 2)
            dia.append(dia_idx_rounded)

    # Convert lists to numpy arrays for element-wise operations
    s1 = np.array(s1)
    s2 = np.array(s2)
    sys = np.round(sys).astype(int)
    dia = np.round(dia).astype(int)  # assuming dia calculation logic


    # # Resample the signal
    # sig = librosa.resample(sig, orig_sr=orig_sr, target_sr=resampling)

    # Scaling factor for the indices
    scale_factor = resampling / orig_sr

    # Adjust indices to the new sample rate
    s1_resampled = np.round(s1 * scale_factor).astype(int)
    s2_resampled = np.round(s2 * scale_factor).astype(int)
    sys_resampled = np.round(sys * scale_factor).astype(int)
    dia_resampled = np.round(dia * scale_factor).astype(int)

    return sig, s1_resampled, s2_resampled, sys_resampled, dia_resampled

##(js)##
a_000 = glob('/home/brody9512/workspace/changhyun/PCG_infer/AMC_heart_sound_labeled_data/*/*.npz')[0]
sig, s1_resampled, s2_resampled, sys_resampled, dia_resampled = load_data_2016_amc(a_000,1000,orig_sr=4000, low=20, high=200, bandpass=True, downsampling=True, up_=None, down_=None)

print(s1_resampled,'\n', sys_resampled,'\n', s2_resampled,'\n', dia_resampled)
seg = {'s1': s1_resampled,'sys': sys_resampled,'s2': s2_resampled,'dia': dia_resampled}
print(seg)



# Initialize the segmentation array with default value 4
seg = np.full_like(sig, 1)

# Assign the specific index positions their respective values
seg[s1_resampled] = 2  # Setting indices from s1_resampled to 1
seg[sys_resampled] = 3  # Setting indices from sys_resampled to 2
seg[s2_resampled] = 4  # Setting indices from s2_resampled to 3
seg[dia_resampled] = 1  # Setting indices from dia_resampled to 4

# Prepare the dictionary to return or use
seg_dict = {'s1': s1_resampled, 'sys': sys_resampled, 's2': s2_resampled, 'dia': dia_resampled, 'seg': seg}

# Example of how to print or use the seg_dict
print("Segmentation array:", seg_dict['seg'][600:650])
print("Lengths: s1 =", len(seg_dict['s1']), "sys =", len(seg_dict['sys']), "s2 =", len(seg_dict['s2']), "dia =", len(seg_dict['dia']))


##(js)##

# Fetch the list of files
a_000 = glob('/home/brody9512/workspace/changhyun/PCG_infer/amc_data/20240417_AMC_labeled_heartsound/*/*.npz')
print(f"Number of files found: {len(a_000)}")
if len(a_000) == 0:
    raise ValueError("No .npz files found. Check your directory path and file type.")

def process_data(low, high, sampling, feature_length):
    total_data = []
    for idx in tqdm.trange(len(a_000)):
        file_path = a_000[idx]
        if os.path.isdir(file_path):
            continue  # Skip directories, just in case
        #print(f"Processing file: {file_path}")  # Debugging line
        try:
            sig, s1_resampled, s2_resampled, sys_resampled, dia_resampled = load_data_2016_amc(file_path, sampling, orig_sr=4000, low=low, high=high, bandpass=True, downsampling=True, up_=None, down_=None)
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        seg = np.full_like(sig, 1)
        seg[dia_resampled] = 1
        seg[s1_resampled] = 2
        seg[sys_resampled] = 3
        seg[s2_resampled] = 4
        total_data.append({'wav': sig, 'dia': dia_resampled, 's1': s1_resampled, 'sys': sys_resampled, 's2': s2_resampled,  'seg': seg, 'fname': file_path.split('/')[-1]})

    total_feature_length_data = [t for t in total_data if len(t['wav']) >= feature_length]
    print(len(total_feature_length_data))
    np.save(f'/home/brody9512/workspace/changhyun/PCG_infer/amc_{sampling}Hz_{low}_{high}_fe_{feature_length}.npy', total_feature_length_data)

parameter_combinations = [(20, 200, 1000, 6144)]
for low_, high_, sampling_, feature_length_ in parameter_combinations:
    process_data(low_, high_, sampling_, feature_length_)