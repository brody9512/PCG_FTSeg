import os
import glob
import numpy as np
import pandas as pd
import librosa as lb
from scipy.signal import resample_poly, butter, lfilter, filtfilt
import natsort
from tqdm import trange

# -------------------- Utility & Filter Functions --------------------
def normalize_signal(signal):
    """
    Normalize the signal to have zero mean and be scaled by its maximum absolute value.
    """
    return (signal - np.mean(signal)) / np.max(np.abs(signal))

def butter_bandpass(lowcut, highcut, fs, order=5):
    """
    Create a Butterworth bandpass filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, fs, lowcut, highcut, order=5):
    """
    Apply a Butterworth bandpass filter to the data.
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return lfilter(b, a, data)

def butter_lowpass_filter(data, cutoff, fs, order=5):
    """
    Apply a Butterworth lowpass filter to the data.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, data)

# -------------------- Data Resampling and Annotation Adjustment --------------------
def resample_audio(wav_path, seg_path, resampling, low=None, high=None, lowpass=None,
                   downsampling=False, bandpass=False, up_=None, down_=None):
    """
    Load, resample, normalize, and filter an audio file while adjusting its segmentation.
    
    Parameters:
        wav_path (str): Path to the audio file (.wav).
        seg_path (str): Path to the segmentation file (.tsv).
        resampling (int): The target sampling rate.
        low (float): Lower cutoff for filtering.
        high (float): Higher cutoff for filtering.
        lowpass (float): Lowpass cutoff (used if bandpass is False).
        downsampling (bool): Whether to downsample the signal.
        bandpass (bool): Whether to apply a bandpass filter.
        up_ (int): Upsampling factor (used with resample_poly).
        down_ (int): Downsampling factor (used with resample_poly).
    
    Returns:
        tuple: (processed_audio, processed_segmentation) or (None, None) if no non-zero segmentation exists.
    """
    # Load audio with original sampling rate
    wav, orig_sr = lb.load(wav_path, sr=None)
    
    if downsampling:
        wav = resample_poly(wav, up=up_, down=down_)
    
    wav = normalize_signal(wav)
    
    if bandpass:
        wav = butter_bandpass_filter(wav, fs=resampling, lowcut=low, highcut=high, order=5)
    else:
        wav = butter_lowpass_filter(wav, cutoff=lowpass, fs=resampling, order=5)
    
    # Load segmentation TSV file (assumes no header and tab-separated)
    df = pd.read_csv(seg_path, sep='\t', header=None)
    seg = np.zeros_like(wav)
    
    # For each row in the TSV, assign the segmentation label
    for idx in range(len(df)):
        start_idx = int(df.iloc[idx, 0] * resampling)
        end_idx = int(df.iloc[idx, 1] * resampling)
        seg[start_idx:end_idx] = df.iloc[idx, 2]
    
    # Trim the signal and segmentation to the non-zero region
    nonzero_indices = np.where(seg != 0)[0]
    if nonzero_indices.size == 0:
        print(f"No non-zero segment found for file {wav_path}")
        return None, None
    wav_new = wav[nonzero_indices[0]:nonzero_indices[-1]]
    seg_new = seg[nonzero_indices[0]:nonzero_indices[-1]]
    
    return wav_new, seg_new

# -------------------- Processing Function --------------------
def process_data_resample(low_, high_, sampling_, feature_length, wav_paths, seg_paths):
    """
    Process the PhysioNet2022 dataset using resampling and filtering parameters,
    and save valid examples as a NumPy file.
    """
    up = sampling_
    down = 4000  # Original sampling rate for PhysioNet2022 is 4000 Hz
    count_with_zeros = 0
    problem_count = 0
    total_resample = []
    
    for i in trange(len(seg_paths)):
        try:
            wav, seg = resample_audio(wav_paths[i], seg_paths[i], resampling=sampling_,
                                      bandpass=True, downsampling=True, low=low_, high=high_, up_=up, down_=down)
            if wav is None or seg is None:
                problem_count += 1
                continue
            # Skip files where segmentation still contains zeros
            if np.any(seg == 0):
                count_with_zeros += 1
                continue
            total_resample.append({
                'wav': wav,
                'seg': seg,
                'fname': os.path.basename(wav_paths[i])
            })
        except Exception as e:
            print(f"Exception occurred for file {wav_paths[i]}: {e}")
            problem_count += 1
    
    print('Total valid files:', len(total_resample))
    print('Files with zeros in segmentation skipped:', count_with_zeros)
    print('Files with problems:', problem_count)
    
    # Filter out examples that do not meet the minimum feature length
    total_new = [t for t in total_resample if len(t['wav']) >= feature_length]
    print('Total files after filtering by feature length:', len(total_new))
    
    output_filename = f'PhysioNet2022_{sampling_}Hz_{low_}_{high_}_fe_{feature_length}.npy'
    np.save(output_filename, total_new)
    print(f"Saved preprocessed PhysioNet2022 data to {output_filename}")

def preprocess_physionet2022_data():
    """
    Preprocess the PhysioNet2022 dataset.
    
    The function expects the following directory structure:
    
        /path/to/PhysioNet2022_DB/*.wav
        /path/to/PhysioNet2022_Modified_TSV/*.tsv
    
    Update the paths below if necessary.
    """
    # Define paths for WAV and TSV files (adjust these paths as needed)
    wav_paths = natsort.natsorted(glob.glob('/home/brody9512/workspace/changhyun/PCG_infer/vital_ch/PhysioNet2022_DB/*.wav'))
    seg_paths = natsort.natsorted(glob.glob('/home/brody9512/workspace/changhyun/PCG_infer/vital_ch/PhysioNet2022_Modified_TSV/*.tsv'))
    
    print(f"Found {len(wav_paths)} WAV files and {len(seg_paths)} segmentation files.")
    
    # Define parameter combinations for filtering and resampling
    resample_params = [
        {'low_': 20, 'high_': 250},
        {'low_': 20, 'high_': 400},
        {'low_': 15, 'high_': 250},
        {'low_': 15, 'high_': 400},
    ]
    
    # Define sampling and feature length values (as tuples: (sampling_rate, feature_length))
    num_values = [(1000, 6144)]
    
    for sampling_, feature_length in num_values:
        print(f'Processing with sampling rate: {sampling_} Hz, feature length: {feature_length}')
        for params in resample_params:
            process_data_resample(**params, sampling_=sampling_, feature_length=feature_length,
                                    wav_paths=wav_paths, seg_paths=seg_paths)

# -------------------- Main Function --------------------
def main():
    print("Starting PhysioNet2022 data preprocessing...")
    preprocess_physionet2022_data()

if __name__ == '__main__':
    main()
