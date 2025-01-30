# jupyter notebook 으로 쓸 때
import multiprocessing
import os, shutil

import numpy as np
import sklearn
import sklearn.metrics
import os
import warnings

import sklearn
import monai
from monai.config import print_config
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import *
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import shutil
from livelossplot import PlotLosses

# Import from Directory Architecture
from config import get_args
from model import *
from modules import *
from dataset import *
from utils import *
from utils import DiceBCELoss, set_seed

# Warnings
warnings.filterwarnings(action='ignore')

# Set environment (optional)
os.environ["HTTP_PROXY"] = "http://192.168.45.100:3128"
os.environ["HTTPS_PROXY"] = "http://192.168.45.100:3128"

args = get_args()

# not_amc=args.not_amc
# not_2022=args.not_2022
# not_2016=args.not_2016

# 인자 값들을 변수에 할당
# featureLength = args.featureLength
# target_sr = args.target_sr
# lowpass = args.lowpass
# year = args.year
# de_conv_=args.de_conv_
# de_fft=args.de_fft
# de_fftconv=args.de_fftconv
# request_infer=args.request_infer
# request_infer_path=args.request_infer_path
# de_aspp = args.de_aspp
# de_deeprft = args.de_deeprft
# de_se= args.de_se
# de_nl= args.de_nl
# de_cbam= args.de_cbam
# not_se=args.not_se
# not_fft=args.not_fft
# se_ratio = args.se_ratio
# dr_se_seq_adverse = args.dr_se_seq_adverse
# dr_se_identity = args.dr_se_identity
# de_dr_se_identity=args.de_dr_se_identity
# twice=args.twice
# third=args.third
# fourth=args.fourth
# conv_=args.conv_
# fft=args.fft
# fftconv=args.fftconv
# nl_ = args.nl
# cbam_ = args.cbam
# seblock_ = args.se
# aspp_ = args.aspp
# deeprft_ = args.deeprft
# mha_=args.mha
# residual_one=args.residual_one
# img_not_residual_one=args.img_not_residual_one
# k_fold_ = args.k_fold
# toler=args.toler
# infer = args.infer
# infer_2022 = args.infer_2022
# nofolder=args.nofolder
# ver = args.ver
# gpus = args.gpu
in_channels = 2
out_channels = 4
minsize=50
thr=0.5
# train_batch= args.batch #64
version=2
max_ep=250
# seed_=args.seed

print(f'pytorch_lightning version : {pl.__version__}')

# featureLength =2560 #1024*2.5
# target_sr=625
# lowpass=240
# year=2016

# aspp_=False
# deeprft_=False
# infer = False

# ver=45
# gpus= "1"


comment=f'ver{version}_d{target_sr}_v{ver}_low{lowpass}_dRFT_{deeprft_}_de_dRFT_{de_deeprft}_aspp_{aspp_}_mha_{mha_}_se_{seblock_}_de_se_{de_se}_nl_{nl_}_cbam_{cbam_}_resione_{residual_one}_twice_{twice}_dr_se_iden_{dr_se_identity}_de_drse_iden_{de_dr_se_identity}_nose_{not_se}_nofft_{not_fft}_drse_seq_ad_{dr_se_seq_adverse}_se_r_{se_ratio}'
load_path=f'/workspace/data/PhysioNet{year}_{target_sr}Hz_{lowpass}_fe_{featureLength}.npy'
infer_pth=f'/workspace/data/lightning_logs/version_{ver}/checkpoints/'

if not infer:
    path=f'/workspace/data/pcg_2016_jupyters/result/{year}_toler{toler}_{comment}/'

    if os.path.exists(path): 
        shutil.rmtree(path) 
    os.mkdir(path)

    data2016=np.load(load_path,allow_pickle=True)

    #data_test2016 = data2016[:3] #training-a data만 긁어옴
    data_test2016 = data2016[:336] #training-a data만 긁어옴
    data_train2016 = data2016[336:]
        
    data_train2016,data_valid2016= sklearn.model_selection.train_test_split(data_train2016,test_size=0.2,random_state=42)
    print(len(data_train2016), len(data_valid2016), len(data_test2016))
    


# gpus= "0,1,2,3"
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["WANDB_API_KEY"] = '6cd6a2f58c8f4625faaea5c73fe110edab2be208'





set_seed(seed_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = os.cpu_count()
print("Number of workers:", NUM_WORKERS)
print('multiprocessing.cpu_count()', multiprocessing.cpu_count())
print('cuda.is_available', torch.cuda.is_available())
print(device)
print_config()


net = BasicUNet(spatial_dims=1, in_channels=in_channels, out_channels=out_channels, features= (64, 64, 128, 256, 512, 512, 64), norm='instance', upsample='pixelshuffle',act='gelu')

if not infer:    
    
    test_ds2016 = PCGDataset(data_test2016)
    test_loader = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)

    train_ds2016 = PCGDataset(data_train2016,'train')
    train_loader = DataLoader(train_ds2016,shuffle=True,batch_size=args.batch,drop_last=True)
    
    valid_ds2016 = PCGDataset(data_valid2016)
    valid_loader = DataLoader(valid_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate) 




plotlosses = PlotLosses()

      
    
lossfn = DiceBCELoss()
yhat = torch.rand(1,3,64)
y = torch.rand(1,3,64).round()
lossfn(yhat,y)





learning_rate= 2e-4
    
model = SEGNET()


trainer = pl.Trainer(
    log_every_n_steps = 1,
    gradient_clip_algorithm='norm',

    
    accumulate_grad_batches=4,

    sync_batchnorm=True,
    benchmark=True,

    accelerator='gpu',
    devices=-1,

    max_epochs=max_ep,

    strategy ='ddp_find_unused_parameters_true',


    check_val_every_n_epoch=1,


    callbacks=[model.checkpoint_callback,LearningRateMonitor(), EarlyStopping('val_loss', patience=20), 
               StochasticWeightAveraging(swa_epoch_start= 0.1, annealing_epochs= 2, swa_lrs=learning_rate*1)],)



if infer:
    ...

else:
    trainer.fit(model, train_loader, valid_loader)




'''tolerence window 40 '''
if not not_2016:
    year = 2016 
    path=f'/workspace/data/pcg_2016_jupyters/result/{year}_toler{toler}_{comment}/'

    if not nofolder:
        if os.path.exists(path): 
            shutil.rmtree(path) 
        os.mkdir(path)

    data2016=np.load(load_path,allow_pickle=True)

    #data_test2016 = data2016[:3] #training-a data만 긁어옴
    data_test2016 = data2016[:336] #training-a data만 긁어옴
    data_train2016 = data2016[336:]
    
    test_ds2016 = PCGDataset(data_test2016)
    test_loader = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)
    
    print('\n ############# toler 40 Internal 2016 start ############# \n')
    checkpoint = torch.load(os.path.join(infer_pth, 'best.ckpt'), map_location='cpu')
    print("Checkpoint state_dict keys:", len(checkpoint['state_dict'].keys()))
    print("Model state_dict keys:", len(model.state_dict().keys()))

    trainer.test(model, test_loader, ckpt_path=os.path.join(infer_pth,'best.ckpt'))







################################
if not not_2022:
    print('\n toler 40 External 2022 start \n')
    year = 2022
    path=f'/workspace/data/pcg_2016_jupyters/result/{year}_toler{toler}_{comment}/'

    infer_pth=f'/workspace/data/lightning_logs/version_{ver}/checkpoints/'

    if not nofolder:
        if os.path.exists(path): 
            shutil.rmtree(path) 
        os.mkdir(path)

    load_path=f'/workspace/data/PhysioNet{year}_{target_sr}Hz_{lowpass}_fe_{featureLength}.npy'
    data2016=np.load(load_path,allow_pickle=True)

    data_test2016=data2016
            
    test_ds2016 = PCGDataset(data_test2016)
    #etest_ds2016 = PCGDataset(edata_test2016)
    test_loader_2022 = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)

    trainer.test(model, test_loader_2022, ckpt_path=os.path.join(infer_pth,'best.ckpt'))








 


model = SEGNET()

trainer = pl.Trainer(
    log_every_n_steps = 1,
    gradient_clip_algorithm='norm',   
    accumulate_grad_batches=4,
    sync_batchnorm=True,
    benchmark=True,
    accelerator='gpu',
    devices=-1,
    max_epochs=max_ep,
    strategy ='ddp_find_unused_parameters_true',
    check_val_every_n_epoch=1,
    callbacks=[model.checkpoint_callback,LearningRateMonitor(), EarlyStopping('val_loss', patience=20), 
               StochasticWeightAveraging(swa_epoch_start= 0.1, annealing_epochs= 2, swa_lrs=learning_rate*1)],)


if not not_amc:
    print('\n toler 40 External amc start \n')
    year = 'amc'
    path=f'/workspace/data/pcg_2016_jupyters/result/{year}_toler{toler}_{comment}/'

    if not nofolder:
        if os.path.exists(path): 
            shutil.rmtree(path) 
        os.mkdir(path)

    load_path=f'/workspace/data/{year}_{target_sr}Hz_{lowpass}_fe_{featureLength}.npy'
    data2016=np.load(load_path,allow_pickle=True)

    data_test2016=data2016
            
    test_ds2016 = PCGDataset(data_test2016)
    test_loader_amc = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)

    trainer.test(model, test_loader_amc, ckpt_path=os.path.join(infer_pth,'best.ckpt'))

