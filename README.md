# PCG_FTSeg: Enhancement of Phonocardiogram Segmentation using Convolutional Neural Networks with Fourier transform Module

<p align="center"><img width="100%" src="./image/model_figure.png" /></p>

## 💡 Highlights
+ Introduce an enhanced U-Net architecture integrated with Convolutional Fourier Transform (CF) modules that fuse time-domain convolution and frequency-domain analysis via FFT and iFFT, enabling precise extraction of both temporal and spectral features for accurate PCG segmentation.
+ Employ a dual CF module strategy—applied consecutively in both encoder and decoder layers—to robustly differentiate between S1 and S2 heart sounds and background noise, thereby significantly improving segmentation performance even in the presence of heart murmurs.
+ Validate the proposed approach on multiple datasets (internal PhysioNet 2016 and external PhysioNet 2022 and AMC), demonstrating superior performance with an average F1 score of 97.64% for S1 and S2 segmentation compared to state-of-the-art methods such as LR-HSMM, LSTM, and BiLSTM.

## 📄 Paper
This repository provides the official implementation code for the paper: **"Enhancement of Phonocardiogram Segmentation using Convolutional Neural Networks with Fourier transform Module."**

Authors: [Changhyun Park](https://github.com/brody9512), Keewon Shin, Jinew Seo, Hyunseok Lim, Gyeong Hoon Kim, Woo-
Young Seo, Sung-Hoon Kim, Namkug Kim

[MI2RL LAB](https://www.mi2rl.co/)

## 💾 Requirements
- numpy >= 1.21.0
- scipy >= 1.7.0
- torch >= 1.10.0
- pytorch-lightning >= 1.6.0
- monai >= 0.9.0
- scikit-learn >= 1.0.0
- neurokit2 >= 0.1.9
- livelossplot >= 0.5.5
- matplotlib >= 3.4.0
- librosa >= 0.10.2
- natsort >= 8.4.0

## 🧱 Repository Structure
```
├── config.py 
├── dataset.py 
├── main.py 
├── model.py 
├── modules.py 
├── utils.py 
├── image 
│ └── model_figure.png 
├── preprocessing 
│ └── signal_2016_and_AMC.py 
│ └── signal_2022.py
```

## 🗂️ Datasets
- Internal Dataset (Train/Validation):
    - Uses the publicly available [PhysioNet 2016](https://physionet.org/content/challenge-2016/1.0.0/) dataset
- External Datasets (Test):
    - Public [PhysioNet 2022](https://physionet.org/content/challenge-2022/1.0.0/) dataset
    - Asan Medical Center (AMC) data; however, due to security regulations in Korea, the AMC data cannot be disclosed

## 📜 Script Example
```bash
python main.py 
    --gpu 0 
    --ver ['appropriate version identifier each time you run the script'] 
    --toler 40 
    --featureLength 6144 
    --target_sr 1000 
    --lowpass 20_200 
    --year 2016 
    --fft 
    --twice
```

## 🎯 Results: 
Comparison of Methods on MITHSDB Benchmark (Tolerance Window = 40 ms)
### Internal (PhysioNet 2016)
| Method                 | Se (%)         | P+ (%)         | F1 S1 (%)       | F1 S2 (%)       | F1 (%)         |
|------------------------|---------------:|---------------:|----------------:|----------------:|---------------:|
| **LR-HSMM**           | 95.6           | 94.4           | -               | -               | 95.0           |
| **LSTM**              | 93.50          | 92.47          | 94.27           | 91.67           | 92.95          |
| **BiLSTM**            | 94.87          | 94.53          | 95.58           | 93.86           | 94.93          |
| **U-Net**             | 91.72±0.10*    | 92.06±0.14*    | 94.72±0.11*     | 88.82±0.06*     | 91.77±0.02*    |
| **U-Net + CFT (Ours)**| 97.67±0.03     | 97.64±0.15     | 98.82±0.16      | 96.46±0.03      | 97.64±0.09     |
### External - 1 (PhysioNet 2022)
| Method                 | Se (%)         | P+ (%)         | F1 S1 (%)       | F1 S2 (%)       | F1 (%)         |
|------------------------|---------------:|---------------:|----------------:|----------------:|---------------:|
| **U-Net**             | 89.74±0.27*    | 91.94±0.20*    | 90.81±0.47*     | 90.54±0.00*     | 90.67±0.24*    |
| **U-Net + CFT (Ours)**| 94.62±0.26     | 95.57±0.01     | 94.97±0.14      | 95.11±0.12      | 95.04±0.13     |
### External - 2 (AMC)
| Method                 | Se (%)         | P+ (%)         | F1 S1 (%)       | F1 S2 (%)       | F1 (%)         |
|------------------------|---------------:|---------------:|----------------:|----------------:|---------------:|
| **U-Net**             | 97.83±0.30*    | 98.33±0.38*    | 98.04±0.42*     | 98.07±0.27*     | 98.06±0.35*    |
| **U-Net + CFT (Ours)**| 99.18±0.13     | 99.16±0.14     | 99.12±0.07      | 99.21±0.02      | 99.16±0.14     |
> **Note:** Data are shown as mean ± standard deviation.  
> `* p-value < 0.05 (paired t-test)`, comparing the proposed model vs. the other methods.  
> The best results are highlighted in **bold**.  
> - **BiLSTM**: Bidirectional long short-term memory  
> - **CF**: Convolutional Fourier transform  
> - **LR-HSMM**: Logistic regression-hidden semi-Markov model  
> - **LSTM**: Long short-term memory  
> - **MITHSDB**: Massachusetts Institute of Technology Heart Sound Database  

## 📝 Citation
If you use this code or find it useful in your research, please cite our paper:
```bibitex
@article{pcg_ftseg,
  title={Enhancement of Phonocardiogram Segmentation using Convolutional Neural Networks with Fourier transform Module},
  author={Changhyun Park and Keewon Shin and Jinew Seo and Hyunseok Lim and Gyeong Hoon Kim and Woo-Young Seo and Sung-Hoon Kim and Namkug Kim},
  year={2025}
}
```

##🙋🏻‍♂️ Acknowledgements
We would like to thank **Junseong Lee** ([@junjslee](https://github.com/junjslee)) for his notable contributions to refactoring the code structure and setting up this GitHub repository.

