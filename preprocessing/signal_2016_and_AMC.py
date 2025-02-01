import os
import glob
import copy
import numpy as np
import scipy
import scipy.io
import librosa as lb
from scipy.signal import butter, lfilter, filtfilt, resample_poly
from tqdm import trange

# -------------------- Filter Functions --------------------
def butter_bandpass(lowcut, highcut, fs, order=5):
    """Create a Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut=25, highcut=400, order=5):
    """Apply a Butterworth bandpass filter to the data."""
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass_filter(data, fs, cutoff, order):
    """Apply a Butterworth lowpass filter to the data."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

# -------------------- Data Loading Functions --------------------
def load_data_2016(wav_path, seg_path, resampling, low=None, high=None, lowpass=None,
                   downsampling=False, bandpass=False):
    """
    Load and process a 2016 wav file and its corresponding segmentation annotation.
    """
    # Load wav file with original sampling rate
    wav, orig_sr = lb.load(wav_path, sr=None)

    # Downsample if requested
    if downsampling:
        wav = resample_poly(wav, up=resampling, down=orig_sr)

    # Apply filtering
    if bandpass:
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        wav = butter_lowpass_filter(wav, fs=resampling, cutoff=lowpass, order=5)

    # Load segmentation from MATLAB file
    df = scipy.io.loadmat(seg_path)['state_ans']
    seg = np.zeros_like(wav)
    for idx in range(len(df) - 1):
        start = int(df[idx, 0][0, 0] * resampling / orig_sr)
        end = int(df[idx + 1, 0][0, 0] * resampling / orig_sr)
        cls = df[idx, 1][0, 0][0]
        if cls == 'diastole':
            cls = 1
        elif cls == 'S1':
            cls = 2
        elif cls == 'systole':
            cls = 3
        elif cls == 'S2':
            cls = 4
        seg[start:end] = int(cls)

    # Trim leading and trailing zeros
    nonzero = np.where(seg != 0)[0]
    if nonzero.size == 0:
        raise ValueError("Segmentation array is empty after filtering.")
    seg = seg[nonzero[0]:nonzero[-1]]
    wav = wav[nonzero[0]:nonzero[-1]]

    return wav, seg

def load_data_amc(sig_path, resampling, orig_sr=4000, low=None, high=None, lowpass=None,
                  downsampling=False, bandpass=False):
    """
    Load and process an AMC npz file containing the signal and segmentation indices.
    """
    data = np.load(sig_path)
    sig = data['sig']
    sig = np.float32(sig / np.max(np.abs(sig)))  # Normalize

    if downsampling:
        sig = resample_poly(sig, up=resampling, down=orig_sr)

    if bandpass:
        sig = butter_bandpass_filter(sig, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        sig = butter_lowpass_filter(sig, fs=resampling, cutoff=lowpass, order=5)

    # Extract segmentation indices
    s1 = data['s1']
    s2 = data['s2']
    sys_list = []
    dia_list = []

    # For systole: for each s1, find the nearest s2 that is not lower than s1
    for i in range(len(s1)):
        s2_candidates = s2[s2 >= s1[i]]
        if len(s2_candidates) > 0:
            s2_closest = s2_candidates[0]
            sys_list.append(round((s1[i] + s2_closest) / 2))

    # For diastole: for each s2, find the nearest s1 that is greater than s2
    for i in range(len(s2)):
        s1_candidates = s1[s1 > s2[i]]
        if len(s1_candidates) > 0:
            s1_closest = s1_candidates[0]
            dia_list.append(round((s2[i] + s1_closest) / 2))

    s1 = np.array(s1)
    s2 = np.array(s2)
    sys_array = np.array(sys_list)
    dia_array = np.array(dia_list)

    # Adjust indices to new sampling rate
    scale_factor = resampling / orig_sr
    s1_resampled = np.round(s1 * scale_factor).astype(int)
    s2_resampled = np.round(s2 * scale_factor).astype(int)
    sys_resampled = np.round(sys_array * scale_factor).astype(int)
    dia_resampled = np.round(dia_array * scale_factor).astype(int)

    return sig, s1_resampled, s2_resampled, sys_resampled, dia_resampled

# -------------------- Preprocessing Functions --------------------
def preprocess_2016_data():
    """
    Preprocess the 2016 data by matching wav files with annotations,
    filtering out mismatches and errors, and saving valid examples.
    """
    # Define directories for audio and annotation files
    data_dirs = [
        './training/training-a',
        './training/training-b',
        './training/training-c',
        './training/training-d',
        './training/training-e',
        './training/training-f'
    ]
    annot_dirs = [
        './annotations/annotations/hand_corrected/training-a_StateAns',
        './annotations/annotations/hand_corrected/training-b_StateAns',
        './annotations/annotations/hand_corrected/training-c_StateAns',
        './annotations/annotations/hand_corrected/training-d_StateAns',
        './annotations/annotations/hand_corrected/training-e_StateAns',
        './annotations/annotations/hand_corrected/training-f_StateAns'
    ]

    wav_files_group = []
    seg_files_group = []

    for i in range(len(data_dirs)):
        wav_files = glob.glob(os.path.join(data_dirs[i], '*.wav'))
        seg_files = glob.glob(os.path.join(annot_dirs[i], '*'))
        print(f"Group {i}: Found {len(wav_files)} wav files and {len(seg_files)} annotation files.")

        # Extract file identifiers
        wav_ids = [os.path.splitext(os.path.basename(s))[0] for s in wav_files]
        seg_ids = [os.path.basename(s).split('_')[0] for s in seg_files]

        # Identify missing annotations
        missing_ids = [num for num in wav_ids if num not in seg_ids]
        print(f"Group {i}: {len(missing_ids)} wav files missing annotations.")

        # Remove wav files that lack a matching annotation
        for mid in missing_ids:
            wav_path = os.path.join(data_dirs[i], f'{mid}.wav')
            if wav_path in wav_files:
                wav_files.remove(wav_path)

        wav_files_group.append(wav_files)
        seg_files_group.append(seg_files)
        print(f"Group {i}: {len(wav_files)} wav files remain after removal.")

    # Flatten and sort lists to (hopefully) ensure matching order
    wav_files_flat = sorted([f for sublist in wav_files_group for f in sublist])
    seg_files_flat = sorted([f for sublist in seg_files_group for f in sublist])

    # Validate that each file pair can be loaded
    valid_wav_files = []
    valid_seg_files = []
    for wav_path, seg_path in zip(wav_files_flat, seg_files_flat):
        try:
            _ = load_data_2016(wav_path, seg_path, resampling=1000, low=20, high=250,
                               bandpass=True, downsampling=True)
            valid_wav_files.append(wav_path)
            valid_seg_files.append(seg_path)
        except Exception as e:
            print(f"Skipping {wav_path} due to error: {e}")

    total_data = []
    for wav_path, seg_path in zip(valid_wav_files, valid_seg_files):
        try:
            wav, seg = load_data_2016(wav_path, seg_path, resampling=1000, low=20, high=250,
                                      bandpass=True, downsampling=True)
            total_data.append({'wav': wav, 'seg': seg, 'fname': os.path.basename(wav_path)})
        except Exception as e:
            print(f"Error processing {wav_path}: {e}")

    # Filter out examples shorter than the desired feature length
    feature_length = 6144  # Adjust this value as needed
    total_feature_length_data = [d for d in total_data if len(d['wav']) >= feature_length]
    print(f"2016 Data: {len(total_feature_length_data)} samples after filtering (feature length >= {feature_length}).")

    output_path = f'PhysioNet2016_1000Hz_20_250_fe_{feature_length}.npy'
    np.save(output_path, total_feature_length_data)
    print(f"Saved preprocessed 2016 data to {output_path}.")

def preprocess_amc_data():
    """
    Preprocess AMC data from .npz files, applying filtering and resampling,
    and save only those examples meeting the feature length criteria.
    """
    # Fetch AMC files
    amc_files = glob.glob('/home/brody9512/workspace/changhyun/PCG_infer/amc_data/20240417_AMC_labeled_heartsound/*/*.npz')
    print(f"Found {len(amc_files)} AMC files.")
    if not amc_files:
        raise ValueError("No .npz files found. Check your directory path and file type.")

    total_data = []
    for idx in trange(len(amc_files)):
        file_path = amc_files[idx]
        if os.path.isdir(file_path):
            continue
        try:
            sig, s1_resampled, s2_resampled, sys_resampled, dia_resampled = load_data_amc(
                file_path, resampling=1000, orig_sr=4000, low=20, high=200,
                bandpass=True, downsampling=True
            )
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            continue

        # Build segmentation array
        seg = np.full_like(sig, 1)
        seg[dia_resampled] = 1 # Note: The AMC code originally set dia indices to 1 (same as default).
        seg[s1_resampled] = 2
        seg[sys_resampled] = 3
        seg[s2_resampled] = 4
        total_data.append({
            'wav': sig,
            'dia': dia_resampled,
            's1': s1_resampled,
            'sys': sys_resampled,
            's2': s2_resampled,
            'seg': seg,
            'fname': os.path.basename(file_path)
        })

    feature_length = 6144  # Adjust this value as needed
    total_feature_length_data = [d for d in total_data if len(d['wav']) >= feature_length]
    print(f"AMC Data: {len(total_feature_length_data)} samples after filtering (feature length >= {feature_length}).")

    output_path = f'amc_1000Hz_20_200_fe_{feature_length}.npy'
    np.save(output_path, total_feature_length_data)
    print(f"Saved preprocessed AMC data to {output_path}.")

# -------------------- Main Function --------------------
def main():
    print("Starting preprocessing for 2016 data...")
    preprocess_2016_data()

    print("\nStarting preprocessing for AMC data...")
    preprocess_amc_data()

if __name__ == '__main__':
    main()
