# config.py
import argparse

def get_args():
    """
    Returns a namespace of parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description='Configure specific Arguments')

    # Hardware / version
    parser.add_argument('--gpu', type=str, default='1')
    parser.add_argument('--ver', type=int, default=45)

    # Feature-related
    parser.add_argument('--featureLength', type=int, default=4096)
    parser.add_argument('--target_sr', type=int, default=1000)
    parser.add_argument('--lowpass', type=str, default=240)
    parser.add_argument('--toler', type=int, default=40)
    parser.add_argument('--year', type=int, default=2016)

    # Model / modules in boolean
    # parser.add_argument('--aspp', action='store_true')
    # parser.add_argument('--deeprft', action='store_true')
    # parser.add_argument('--se', action='store_true') # conv 축소하고 압축시키는 방법 sqeeze and excitation
    # parser.add_argument('--nl', action='store_true') # 위와 비슷한 기법 
    # parser.add_argument('--cbam', action='store_true')
    # parser.add_argument('--mha', action='store_true') # multihead attention
    
    # Inference / folder
    parser.add_argument('--infer', action='store_true') # 2016
    parser.add_argument('--infer_2022', action='store_true')
    parser.add_argument('--request_infer', action='store_true')
    parser.add_argument('--request_infer_path', type=str)
    parser.add_argument('--nofolder', action='store_true')
    # parser.add_argument('--k_fold', action='store_true')
    
    # Additional SE / RFT flags
    # parser.add_argument('--se_ratio', type=int, default=2)
    # parser.add_argument('--dr_se_seq_adverse', action='store_true')
    # parser.add_argument('--dr_se_identity', action='store_true')
    parser.add_argument('--img_not_residual_one', action='store_true') # 혹시 모르니 보류 
    # parser.add_argument('--residual_one', action='store_true')
    # parser.add_argument('--de_aspp', action='store_true')
    # parser.add_argument('--de_deeprft', action='store_true')
    # parser.add_argument('--de_se', action='store_true')
    # parser.add_argument('--de_nl', action='store_true')
    # parser.add_argument('--de_cbam', action='store_true')
    # parser.add_argument('--de_dr_se_identity', action='store_true')
    # parser.add_argument('--not_se', action='store_true')
    parser.add_argument('--not_fft', action='store_true')
    parser.add_argument('--twice', action='store_true')
    # parser.add_argument('--third', action='store_true')
    # parser.add_argument('--fourth', action='store_true')
    parser.add_argument('--conv_', action='store_true')
    parser.add_argument('--fft', action='store_true')
    # parser.add_argument('--fftconv', action='store_true')
    parser.add_argument('--de_conv_', action='store_true')
    parser.add_argument('--de_fft', action='store_true')
    # parser.add_argument('--de_fftconv', action='store_true')
    parser.add_argument('--batch', type=int, default=64)
    
    # Dataset usage toggles
    parser.add_argument('--not_2016', action='store_true')
    parser.add_argument('--not_2022', action='store_true')
    parser.add_argument('--not_amc', action='store_true')
    
    # Seed
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()
