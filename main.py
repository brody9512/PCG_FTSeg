
# jupyter notebook 으로 쓸 때
import multiprocessing
# multiprocessing.set_start_method('spawn', force=True)
import os, sys, shutil
#import multiprocessing

from model import *
from modules import *
from data import *
from utils import *

from utils import DiceBCELoss, set_seed

import pylab as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sklearn
import sklearn.metrics

import os
os.environ["HTTP_PROXY"] = "http://192.168.45.100:3128"
os.environ["HTTPS_PROXY"] = "http://192.168.45.100:3128"

import librosa as lb
import pylab as plt
import glob
import natsort
import scipy
import datetime

#%matplotlib inline
import warnings
warnings.filterwarnings(action='ignore')


import matplotlib.colors as mcolors
from glob import glob
from tqdm.notebook import tqdm, trange
from natsort import natsorted

import scipy
import scipy.io as sio
from skimage import morphology
from scipy import ndimage
from utils import *

import kornia
import neurokit2 as nk
import librosa as lb
import librosa.display
import soundfile as sf

import sklearn

import cv2
import monai
from monai.inferers import sliding_window_inference
from monai.config import print_config

import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import skimage
import skimage.morphology
from typing import Optional, Sequence, Tuple, Union
import shutil
import math
import argparse

from matplotlib import gridspec

from livelossplot import PlotLosses
from parser import get_parser

parser = get_parser()

args = parser.parse_args()

not_amc=args.not_amc
not_2022=args.not_2022
not_2016=args.not_2016

# 인자 값들을 변수에 할당
featureLength = args.featureLength
target_sr = args.target_sr
lowpass = args.lowpass
year = args.year


de_conv_=args.de_conv_
de_fft=args.de_fft
de_fftconv=args.de_fftconv

request_infer=args.request_infer
request_infer_path=args.request_infer_path

de_aspp = args.de_aspp
de_deeprft = args.de_deeprft
de_se= args.de_se
de_nl= args.de_nl
de_cbam= args.de_cbam

not_se=args.not_se
not_fft=args.not_fft

se_ratio = args.se_ratio
dr_se_seq_adverse = args.dr_se_seq_adverse
dr_se_identity = args.dr_se_identity


de_dr_se_identity=args.de_dr_se_identity

twice=args.twice
third=args.third
fourth=args.fourth


conv_=args.conv_
fft=args.fft
fftconv=args.fftconv

nl_ = args.nl
cbam_ = args.cbam
#sa_ = args.sa

seblock_ = args.se
aspp_ = args.aspp
deeprft_ = args.deeprft
mha_=args.mha

residual_one=args.residual_one
img_not_residual_one=args.img_not_residual_one


k_fold_ = args.k_fold

toler=args.toler

infer = args.infer
infer_2022 = args.infer_2022
nofolder=args.nofolder

ver = args.ver
gpus = args.gpu

# featureLength = 12288
in_channels = 2
out_channels = 4
minsize=50
thr=0.5

train_batch= args.batch #64
version=2

max_ep=250

seed_=args.seed

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
#os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = gpus
os.environ["WANDB_API_KEY"] = '6cd6a2f58c8f4625faaea5c73fe110edab2be208'
#%env WANDB_SILENT=true




set_seed(seed_)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_WORKERS = os.cpu_count()
print("Number of workers:", NUM_WORKERS)
print('multiprocessing.cpu_count()', multiprocessing.cpu_count())
print('cuda.is_available', torch.cuda.is_available())
print(device)
print_config()





from typing import Sequence, Union

from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat, Pool
from monai.utils import ensure_tuple_rep


from typing import Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn

from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import same_padding
from monai.networks.layers.factories import Conv
from monai.networks.blocks import UpSample



def _upsample_like(src,tar):
    src = F.upsample(src,size=tar.shape[2:],mode='linear')

    return src


net = BasicUNet(spatial_dims=1, in_channels=in_channels, out_channels=out_channels, features= (64, 64, 128, 256, 512, 512, 64), norm='instance', upsample='pixelshuffle',act='gelu')

if not infer:    
    
    test_ds2016 = dataset(data_test2016)
    test_loader = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)

    train_ds2016 = dataset(data_train2016,'train')
    train_loader = DataLoader(train_ds2016,shuffle=True,batch_size=train_batch,drop_last=True)
    
    valid_ds2016 = dataset(data_valid2016)
    valid_loader = DataLoader(valid_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate) 
    # collate_fn 쓰는 이유: dataset이 고정된 길이가 아닐 경우, 
    # batchsize를 2 이상으로 dataloader를 호출하면 dataloader에서 batch로 바로 못묶이고 에러가 난다
    #collate_fn()은 variable-length input을 batch로 잘 묶어서 dataloader로 넘겨주는 역할을 한다.






plotlosses = PlotLosses()

      
    
lossfn = DiceBCELoss()
yhat = torch.rand(1,3,64)
y = torch.rand(1,3,64).round()
lossfn(yhat,y)


import torchmetrics


learning_rate= 2e-4
    
model = SEGNET()


trainer = pl.Trainer(
    log_every_n_steps = 1,
    gradient_clip_algorithm='norm',
    # to prevent the gradients from becoming too large, which can cause numerical instability and poor model performance
    # the gradients will be clipped based on their L2-norm (Euclidean norm).
    
    accumulate_grad_batches=4,
    # In this case, gradients will be accumulated for 4 batches before updating the model weights. 
    # This can be useful for training with larger effective batch sizes when GPU memory is limited. 

    sync_batchnorm=True,
    benchmark=True,

    accelerator='gpu',
    devices=-1,
    #When set to -1, it means that all available devices (GPUs) will be used.
    
    #plugins=[DDPPlugin(find_unused_parameters=True)],  # Here's the change for find_unused_parameters
    
    max_epochs=max_ep,

    strategy ='ddp_find_unused_parameters_true',
    #strategy = DDPStrategy(find_unused_parameters=True)  
    
    #'ddp',
    #strategy ='ddp_notebook',
    
    #strategy ='ddp', #파이썬 스크립트로 쓸 때는 이걸로!!
    
    # Each GPU processes its part of the mini-batch, 
    # and the gradients are then averaged across GPUs before updating the model weights.

    check_val_every_n_epoch=1,
    # validation will be performed every 1 epoch, 
    # meaning that it will be run after every complete pass through the training dataset


    callbacks=[model.checkpoint_callback,LearningRateMonitor(), EarlyStopping('val_loss', patience=20), 
               StochasticWeightAveraging(swa_epoch_start= 0.1, annealing_epochs= 2, swa_lrs=learning_rate*1)],)



if infer:
    ...

else:
    trainer.fit(model, train_loader, valid_loader)
#        trainer.save_checkpoint(f'{path}model_{year}_toler{toler}_{comment}.ckpt')




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
    
    test_ds2016 = dataset(data_test2016)
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
            
    test_ds2016 = dataset(data_test2016)
    #etest_ds2016 = dataset(edata_test2016)
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
            
    test_ds2016 = dataset(data_test2016)
    test_loader_amc = DataLoader(test_ds2016,batch_size=1,collate_fn=monai.data.utils.default_collate)

    trainer.test(model, test_loader_amc, ckpt_path=os.path.join(infer_pth,'best.ckpt'))

