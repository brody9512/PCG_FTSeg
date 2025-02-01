import argparse

def get_args():
    parser = argparse.ArgumentParser(description='Configure specific Arguments')

    # GPU / version
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--ver', type=int, default=45)

    # Feature-related
    parser.add_argument('--featureLength', type=int, default=4096)
    parser.add_argument('--target_sr', type=int, default=1000)
    parser.add_argument('--lowpass', type=str, default=240)
    parser.add_argument('--toler', type=int, default=40)
    parser.add_argument('--year', type=int, default=2016)
    
    # Folder
    parser.add_argument('--infer', action='store_true') # 2016
    parser.add_argument('--infer_2022', action='store_true')
    parser.add_argument('--nofolder', action='store_true')
    
    # Modeling
    # parser.add_argument('--not_fft', action='store_true')
    parser.add_argument('--twice', action='store_true')
    parser.add_argument('--conv_', action='store_true')
    parser.add_argument('--fft', action='store_true')
    parser.add_argument('--batch', type=int, default=64)
    
    # Dataset usage
    parser.add_argument('--not_2016', action='store_true')
    parser.add_argument('--not_2022', action='store_true')
    parser.add_argument('--not_amc', action='store_true')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()
