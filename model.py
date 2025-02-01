import datetime
from monai.networks.layers.factories import Conv
from monai.networks.nets.basic_unet import Down, TwoConv, UpCat, Pool
from monai.utils import ensure_tuple_rep
from monai.networks.blocks import UpSample
from monai.inferers import sliding_window_inference
from typing import Optional, Sequence, Union
import numpy as np
import scipy
import skimage
import torch
import torch.nn as nn
import math
import torchmetrics
import pylab as plt
import pytorch_lightning as pl
from pytorch_lightning.callbacks import *
from pytorch_lightning.loggers import *

# Local imports
from modules import BasicConv, fftRFT
from utils import DiceBCELoss, eval_metrics


#
# ------------------ ENCODER BLOCK ------------------
#
class Down(nn.Sequential):
    """maxpooling downsampling and two convolutions."""
    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        sa = False,  # This is a default argument in __init__
        
        twice=False,
        fft=False   
    ):
        super().__init__()
        self.max_pooling = Pool["MAX", spatial_dims](kernel_size=2)
        self.fftRFT_=fft 
        self.sa = sa 
        self.twice = twice
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        
        # Convolutions
        self.convs_init = TwoConv(spatial_dims, in_chns, in_chns, act, norm, bias, dropout)
        self.convs = TwoConv(spatial_dims, in_chns, out_chns, act, norm, bias, dropout)
        
        # Fourier Transform module
        self.fftRFT=fftRFT(in_chns,in_chns)
            
    def get_positional_encoding(self, pos, i, dim):
        angles = 1 / math.pow(10000, (2 * (i // 2))/ dim)

        if i % 2 == 0:
            return math.sin(pos * angles)
        return math.cos(pos * angles)

    def position_encoding(self, x, seq_length, batch_size, device):
        pe = torch.zeros(seq_length, batch_size, self.d_model, device=device)

        for i in range(seq_length):
            for j in range(self.d_model):
                pe[i, :, j] = self.get_positional_encoding(i, j, self.d_model)    
        
        return pe
        
    def forward(self, x: torch.Tensor):
        
        x= self.max_pooling(x)
        
        if self.fftRFT_:
            x = self.fftRFT(x)
        if self.twice:
            x = self.convs_init(x)
            if self.fftRFT_:
                x = self.fftRFT(x)
                
        x= self.convs(x)
        
        return x
    
#
# ------------------  DECODER BLOCK ------------------
#
class UpCat(nn.Module):
    """upsampling, concatenation with the encoder feature map, two convolutions"""

    def __init__(
        self,
        spatial_dims: int,
        in_chns: int,
        cat_chns: int,
        out_chns: int,
        act: Union[str, tuple],
        norm: Union[str, tuple],
        bias: bool,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        pre_conv: Optional[Union[nn.Module, str]] = "default", # do we need this?
        interp_mode: str = "linear",
        align_corners: Optional[bool] = True,
        halves: bool = True,
        is_pad: bool = True,

        twice=False,
        fft=False,
    ):

        super().__init__()
        if upsample == "nontrainable" and pre_conv is None:
            up_chns = in_chns
        else:
            up_chns = in_chns // 2 if halves else in_chns
            
        self.is_pad = is_pad
        self.fftRFT_=fft
        self.twice=twice

        self.upsample = UpSample(
            spatial_dims,
            in_channels=in_chns,
            out_channels=up_chns,
            scale_factor=2,
            mode=upsample,
            pre_conv=pre_conv,
            interp_mode=interp_mode,
            align_corners=align_corners,
        )
        
        
        # TwoConv blocks
        self.convs_init = TwoConv(spatial_dims, cat_chns + up_chns, cat_chns + up_chns, act, norm, bias, dropout) #1024        
        self.convs = TwoConv(spatial_dims, cat_chns + up_chns, out_chns, act, norm, bias, dropout) #1024
        
        # Fourier Transform module
        self.fftRFT=fftRFT(cat_chns + up_chns, cat_chns + up_chns)

    def forward(self, x: torch.Tensor, x_e: Optional[torch.Tensor]):
        """
            x: features to be upsampled.
            x_e: features from the encoder.
        """
        x_0 = self.upsample(x)

        if x_e is not None:
            if self.is_pad:
                # handling spatial shapes due to the 2x maxpooling with odd edge lengths.
                dimensions = len(x.shape) - 2
                sp = [0] * (dimensions * 2)
                for i in range(dimensions):
                    if x_e.shape[-i - 1] != x_0.shape[-i - 1]:
                        sp[i * 2 + 1] = 1
                x_0 = torch.nn.functional.pad(x_0, sp, "replicate")

            x = torch.cat([x_e, x_0], dim=1)

            if self.fft_:
                x = self.fftRFT(x)

            if self.twice:
                x = self.convs_init(x)
                if self.fft_:
                    x = self.fftRFT(x)

            x= self.convs(x)
        else:
            x = self.convs(x_0)

        return x
    
#
# ------------------ BASIC UNET ------------------
#
class BasicUNet(nn.Module):
    def __init__(
        self,
        spatial_dims: int = 3,
        in_channels: int = 1,
        out_channels: int = 4,
        features: Sequence[int] = (32, 32, 64, 128, 256, 32),
        act: Union[str, tuple] = "gelu",
        norm: Union[str, tuple] = ("instance", {"affine": True}),
        bias: bool = True,
        dropout: Union[float, tuple] = 0.0,
        upsample: str = "deconv",
        fft: bool = False,
        twice: bool = False,
    ):
        
        super().__init__()

        fea = ensure_tuple_rep(features, 7)
        print(f"BasicUNet features: {fea}.")
        
        # Initial 2-conv
        self.conv_0 = TwoConv(spatial_dims, in_channels, fea[0], act, norm, bias, dropout)

        # Down blocks
        self.down_1 = Down(spatial_dims, fea[0], fea[1], act, norm, bias, dropout, fft=fft, twice=twice)
        self.down_2 = Down(spatial_dims, fea[1], fea[2], act, norm, bias, dropout, fft=fft, twice=twice)
        self.down_3 = Down(spatial_dims, fea[2], fea[3], act, norm, bias, dropout, fft=fft, twice=twice)
        self.down_4 = Down(spatial_dims, fea[3], fea[4], act, norm, bias, dropout, fft=fft, twice=twice)
        self.down_5 = Down(spatial_dims, fea[4], fea[5], act, norm, bias, dropout, fft=fft, twice=twice)

        # Up blocks
        self.upcat_5 = UpCat(spatial_dims, fea[5], fea[4], fea[4], act, norm, bias, dropout, upsample, fft=fft, twice=twice)
        self.upcat_4 = UpCat(spatial_dims, fea[4], fea[3], fea[3], act, norm, bias, dropout, upsample, fft=fft, twice=twice)
        self.upcat_3 = UpCat(spatial_dims, fea[3], fea[2], fea[2], act, norm, bias, dropout, upsample, fft=fft, twice=twice)
        self.upcat_2 = UpCat(spatial_dims, fea[2], fea[1], fea[1], act, norm, bias, dropout, upsample, fft=fft, twice=twice)
        # final up path to fea[6]
        self.upcat_1 = UpCat(spatial_dims, fea[1], fea[0], fea[6], act, norm, bias, dropout, upsample, fft=fft, twice=twice, halves=False)

        # Final
        self.final_conv = Conv["conv", spatial_dims](fea[6], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor):
        # First TwoConv
        x0 = self.conv_0(x) 

        x1 = self.down_1(x0)
        x2 = self.down_2(x1)
        x3 = self.down_3(x2)
        x4 = self.down_4(x3)
        x5 = self.down_5(x4)

        u5 = self.upcat_5(x5, x4)
        u4 = self.upcat_4(u5, x3)
        u3 = self.upcat_3(u4, x2)
        u2 = self.upcat_2(u3, x1)
        u1 = self.upcat_1(u2, x0)


        logits = self.final_conv(u1)
        # Temperature scaling or direct sigmoid
        logits = torch.sigmoid(logits/.2)
        return logits
    
#
# ------------------ LIGHTNING MODULE WRAPPER ------------------
#
class SEGNET(pl.LightningModule):
    """
    LightningModule to wrap the BasicUNet (or other nets),
    handle training/validation/test loops, and compute metrics.
    Certain references like 'year', 'path', 'toler', etc. are turned
    into constructor arguments so main.py can set them.
    """
    def __init__(self,
                 net: nn.Module,
                 featureLength: int = 2560,
                 learning_rate: float = 1e-4,
                 in_channels: int = 2,
                 out_channels: int = 4,
                 minsize: int = 50,
                 thr: float = 0.5,
                 device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 # placeholders for the old code that used path/infer_pth/etc.
                 infer_path: str = "/workspace/data/lightning_logs/version_{ver}/checkpoints/",
                 path: str = "./",
                 year: int = 2000,
                 toler: int = 40,
         ):
        super(SEGNET, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.featureLength = featureLength
        self.learning_rate = learning_rate
        self.net = net
        self.minsize = minsize
        self.thr = thr
        self.device = device
        self.toler = toler
        self.infer_path = infer_path
        self.path = path
        self.year = year
        
        self.validation_step_outputs = []
        self.test_step_outputs=[]
        
        self.lossfn = DiceBCELoss()

        self.checkpoint_callback=ModelCheckpoint(monitor='val_loss', mode='min', filename='best', save_top_k=1,
                                      save_last=True,  # Save the latest checkpoint
                                      dirpath=self.infer_pth)
        
       
        if out_channels==5:

            task = 'multilabel'   
            # multilabel: 하나의 그림이 하나의 카테고리(2개 이상)에 속하지 않고, 두 개 이상의 카테고리에 속하게 된다.   
            average = 'micro'
            # micro: Sum statistics over all labels
            self.validACC = torchmetrics.Accuracy(task=task,num_labels=out_channels, average=average)
            self.validSEN = torchmetrics.Recall(task=task,num_labels=out_channels,average=average)
            self.validPPV = torchmetrics.Precision(task=task,num_labels=out_channels,average=average)
            self.validF1 = torchmetrics.F1Score(task=task,num_labels=out_channels,average=average)
                    
        elif out_channels>3: #1234 여야
            task = 'multiclass'
            #Multi-class 분류 문제는 하나의 그림에 하나의 객체만 있어야 하고, 그리고 그 객체는 2개 이상의 카테고리에 속하는 경우
            average = 'macro'
            # macro: Calculate statistics for each label and average them
            # weighted: Calculates statistics for each label and computes weighted average using their support
            self.validACC = torchmetrics.Accuracy(task=task,num_classes=out_channels,average=average)
            self.validSEN = torchmetrics.Recall(task=task,num_classes=out_channels,average=average)
            self.validPPV = torchmetrics.Precision(task=task,num_classes=out_channels,average=average)
            self.validF1 = torchmetrics.F1Score(task=task,num_classes=out_channels,average=average)

        self.save_hyperparameters()

    def compute_loss(self, yhat, y):
        if isinstance(yhat, (tuple, list)):
            return sum(self.lossfn(pred, y) for pred in yhat)
        else:
            return self.lossfn(yhat, y)
    
    def forward(self, x):
        return self.net(x)
        
    def sw_inference(self, x): # Inference Sliding window using MONAI API: Using this only valid and test when size of input is larger than 2048
        return sliding_window_inference(
            inputs=x,
            roi_size=self.featureLength,
            sw_batch_size=16,
            predictor=self.net,
            mode='gaussian',
            overlap=0.25,
            device=self.device,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.5)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }
    #
    # -------------- PIPELINE & STEPS --------------
    #
    def pipeline(self, batch, batch_idx, save=False):
        """
        Shared logic for train/val/test steps
        """
        global keys_
        global metrics
        
        # For custom metrics
        keys = ["s1", "s2", "sys", "dia"]
        sub_keys = ["TP", "FN", "FP", "sen", "pre", "f1"]
        metrics = {key: {sub_key: 0 for sub_key in sub_keys} for key in keys}
        metrics["fname_f1_score"] = {}   

        x = batch['x'].float()
        y = batch['y_onehot'].float()
        fname= batch['fname']

        yhat = self.forward(x) if x.shape[-1] == self.featureLength else self.sw_inference(x)       

        loss = self.compute_loss(yhat,y)
        uncert = torch.zeros_like(yhat[0])
        
        y = y
        yhat = yhat
        
        # If 'save' is True, do your plotting or metrics saving
        if save:
            
                for i in range(len(x)):
                    
                    y_s1 = y[i,0].cpu().detach().numpy().round()
                    idxs_y_s1 = self.get_Binaryindex(y_s1)

                    y_s2 = y[i,2].cpu().detach().numpy().round()
                    idxs_y_s2 = self.get_Binaryindex(y_s2)

                    y_sys = y[i,1].cpu().detach().numpy().round()
                    idxs_y_sys = self.get_Binaryindex(y_sys)

                    y_dia = y[i,3].cpu().detach().numpy().round()
                    idxs_y_dia = self.get_Binaryindex(y_dia)

                    output_all = yhat[i].cpu().detach().numpy().round()
                    output_all = output_all.astype(bool)


                    final_output_all = np.zeros_like(output_all)


                    ##여기에 넘파이 제로를 만들어서 
                    for q in range(len(output_all)):
                        output_all_=skimage.morphology.remove_small_objects(output_all[q], self.minsize, connectivity=1).astype(int)

                        final_output_all[q]=output_all_.astype('int')
                    
                    output_s1 = self.apply_threshold(yhat[i,0].cpu().detach().numpy(),0.5)
                    output_s1 = self.postprocess(output_s1)
                    idxs_yhat_s1 = self.get_Binaryindex(output_s1)

                    output_s2_ = self.apply_threshold(yhat[i,2].cpu().detach().numpy(),self.thr)
                    output_s2 = self.postprocess(output_s2_)
                    idxs_yhat_s2 = self.get_Binaryindex(output_s2)

                    idxs_yhat_s2_ = self.get_Binaryindex(output_s2_)

                    output_sys = yhat[i,1].cpu().detach().numpy().round()
                    output_sys = self.postprocess(output_sys)
                    idxs_yhat_sys = self.get_Binaryindex(output_sys)

                    output_dia = yhat[i,3].cpu().detach().numpy().round()
                    output_dia = self.postprocess(output_dia)
                    idxs_yhat_dia = self.get_Binaryindex(output_dia)

                    segment_names = ["s1", "s2", "sys", "dia"]
                    idxs_y_list = [idxs_y_s1, idxs_y_s2, idxs_y_sys, idxs_y_dia]
                    idxs_yhat_list = [idxs_yhat_s1, idxs_yhat_s2, idxs_yhat_sys, idxs_yhat_dia]

                    for w, seg_name in enumerate(segment_names):
                        TP, FN, FP, sen, pre, f1 = eval_metrics(
                            idxs_y_list[w], idxs_yhat_list[w],
                            metrics[seg_name]["TP"], metrics[seg_name]["FN"], metrics[seg_name]["FP"],
                            metrics[seg_name]["sen"], metrics[seg_name]["pre"], metrics[seg_name]["f1"], self.toler
                        )
                        
                        metrics[seg_name].update({
                            "TP": TP,
                            "FN": FN,
                            "FP": FP,
                            "sen": sen,
                            "pre": pre,
                            "f1": f1
                        })

                    f1_for_current_fname_s1 = 2 * (metrics["s1"]["pre"] * metrics["s1"]["sen"]) / (metrics["s1"]["pre"] + metrics["s1"]["sen"]) if (metrics["s1"]["pre"] + metrics["s1"]["sen"]) != 0 else 0
                    f1_for_current_fname_s2 = 2 * (metrics["s2"]["pre"] * metrics["s2"]["sen"]) / (metrics["s2"]["pre"] + metrics["s2"]["sen"]) if (metrics["s2"]["pre"] + metrics["s2"]["sen"]) != 0 else 0
                    f1_for_current_fname_sys = 2 * (metrics["sys"]["pre"] * metrics["sys"]["sen"]) / (metrics["sys"]["pre"] + metrics["sys"]["sen"]) if (metrics["sys"]["pre"] + metrics["sys"]["sen"]) != 0 else 0
                    f1_for_current_fname_dia = 2 * (metrics["dia"]["pre"] * metrics["dia"]["sen"]) / (metrics["dia"]["pre"] + metrics["dia"]["sen"]) if (metrics["dia"]["pre"] + metrics["dia"]["sen"]) != 0 else 0

                    metrics["fname_f1_score"][fname[i]] = {
                        "s1": round(f1_for_current_fname_s1, 5),
                        "s2": round(f1_for_current_fname_s2, 5),
                        "sys": round(f1_for_current_fname_sys, 5),
                        "dia": round(f1_for_current_fname_dia, 5),
                        "mean": round(np.mean([f1_for_current_fname_s1, f1_for_current_fname_s2, f1_for_current_fname_sys, f1_for_current_fname_dia]), 5)
                    }

                    plt.figure(figsize=(27,35))
                    plt.subplot(611)
                    plt.title(f'{fname[i]}, ground truth')
                    plt.plot(x[i,0].cpu().detach().numpy(),label='x',color='black',alpha=.6)
                    plt.scatter(idxs_y_s1,[5]*len(idxs_y_s1),color='r',label='y_s1')
                    plt.scatter(idxs_y_s2,[5]*len(idxs_y_s2),color='b',label='y_s2')
                    #plt.legend(fontsize=3)
                    plt.subplot(612)
                    plt.title(f'Prediction result')
                    plt.plot(x[i,0].cpu().detach().numpy(),label='x',color='black',alpha=.6)
                    plt.scatter(idxs_yhat_s1,[5]*len(idxs_yhat_s1),color='r',label='yhat_s1')
                    plt.scatter(idxs_yhat_s2,[5]*len(idxs_yhat_s2),color='b',label='yhat_s2')
                    #plt.legend(fontsize=3)
                                        
                    plt.subplot(613)
                    plt.title(f'Ground Truth fill')
                    x_values = x[i, 0].cpu().detach().numpy()
                    output_s1 = y_s1 * np.max(x_values) // 1
                    output_s2 = y_s2 * np.max(x_values) // 1

                    plt.plot(x_values, label='x', color='black', alpha=.6)
                    plt.fill_between(np.arange(len(x_values)), 0, np.where(output_s1 != 0, x_values, np.nan), color='r', alpha=0.5)
                    plt.fill_between(np.arange(len(x_values)), 0, np.where(output_s2 != 0, x_values, np.nan), color='b', alpha=0.5)
                
                    
                    # Ground truth tile
                    sig_alpha = 0.6
                    s1_s2_alpha_gt = 0.5
                    s1_s2_alpha = 0.5

                    # Ground truth tile
                    plt.subplot(614)
                    plt.title('Ground Truth tile')
                    x_values = x[i, 0].cpu().detach().numpy()
                    plt.plot(x_values, label='x', color='black', alpha=sig_alpha)
                    y_s1_map = np.tile(y_s1, (50, 1))
                    y_s2_map = np.tile(y_s2, (50, 1))

                    # Applying the masked values to the ground truth subplot
                    plt.imshow(y_s1_map, aspect='auto', cmap='Reds', alpha=s1_s2_alpha_gt, extent=[0, len(y_s1), np.min(x_values), np.max(x_values)])
                    plt.imshow(y_s2_map, aspect='auto', cmap='Blues', alpha=s1_s2_alpha_gt, extent=[0, len(y_s2), np.min(x_values), np.max(x_values)])
                    
                    
                    plt.subplot(615)
                    plt.title(f'Prediction result (s1)')
                    plt.plot(x[i,0].cpu().detach().numpy(),label='x',color='black',alpha=sig_alpha)
                    yhat_values_s1 = yhat[i, 0].cpu().detach().numpy()
                    yhat_values_s1_masked = np.ma.masked_where(yhat_values_s1 == 0, yhat_values_s1)
                    plt.imshow(np.tile(yhat_values_s1_masked, (50, 1)), aspect='auto', cmap='Reds', alpha=s1_s2_alpha, extent=[0, len(yhat_values_s1), np.min(x[i, 0].cpu().detach().numpy()), np.max(x[i, 0].cpu().detach().numpy())])



                    plt.subplot(616)
                    plt.title(f'Prediction result (s2)')
                    plt.plot(x[i,0].cpu().detach().numpy(),label='x',color='black',alpha=sig_alpha)
                    yhat_values_s2 = yhat[i, 2].cpu().detach().numpy()
                    yhat_values_s2_masked = np.ma.masked_where(yhat_values_s2 == 0, yhat_values_s2)
                    plt.imshow(np.tile(yhat_values_s2_masked, (50, 1)), aspect='auto', cmap='Blues', alpha=s1_s2_alpha, extent=[0, len(yhat_values_s2), np.min(x[i, 0].cpu().detach().numpy()), np.max(x[i, 0].cpu().detach().numpy())])


                    plt.savefig(f'{self.path}z_{fname[i]}.png', dpi=300)
                    plt.close()

        return {'loss':loss, "x": x, "y": y, "yhat":yhat, 'fname':fname[0], 'uncert':uncert, 'metrics':metrics}
    
    def training_step(self, batch, batch_idx):        
        result = self.pipeline(batch, batch_idx)
        self.log('loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True)
        
        return {"loss":result['loss'], "x": result['x'], "y": result['y'], "yhat":result['yhat'], "uncert":result['uncert'],'fname':result['fname']}


    def validation_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx)
        self.log('val_loss', result['loss'], on_step=True, on_epoch=True, prog_bar=True)
        self.validation_step_outputs.append(result)
        
        return result
    
    def on_validation_epoch_end(self):
        self.checkpoint_callback.on_validation_epoch_end(self.current_epoch, self.trainer)
        self.evaluations(self.validation_step_outputs, plot=False, save=False)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        result = self.pipeline(batch, batch_idx,True)
        self.log('test_loss', result['loss'], on_step=False, on_epoch=True)
        self.test_step_outputs.append(result)

        return result
    
    def on_test_epoch_end(self):
        print("test_step_end")        
        self.evaluations(self.test_step_outputs, plot=False, save=True)
        self.test_step_outputs.clear()
        
    #
    # -------------- METRICS & EVAL --------------
    #
    def evaluations(self, outputs, plot=False, save=False):

        keyss = ["s1", "s2", "sys", "dia"]
        sub_keyss = ["TP", "FN", "FP", "sen", "pre", "f1"]

        metrics_ = {keyss: {sub_keyss: 0 for sub_keyss in sub_keyss} for keyss in keyss}        

        paired_test = {}
        paired_precision = {}
        paired_sensitivity = {}

        precision_collection = {"s1": [], "s2": [], "sys": [], "dia": []}
        sensitivity_collection = {"s1": [], "s2": [], "sys": [], "dia": []}

        for o in outputs:
            x = o['x']
            yhat = o['yhat']
            fname = o['fname']
            uncert = o['uncert']
            metrics = o['metrics']          
            
            if save:
                paired_test[fname] =o["metrics"]["fname_f1_score"][fname]

                ## Add precision, sensitivity
                paired_precision[fname] = {
                    "s1": round(metrics["s1"]["pre"], 5),
                    "s2": round(metrics["s2"]["pre"], 5),
                    "sys": round(metrics["sys"]["pre"], 5),
                    "dia": round(metrics["dia"]["pre"], 5)
                }
                paired_sensitivity[fname] = {
                    "s1": round(metrics["s1"]["sen"], 5),
                    "s2": round(metrics["s2"]["sen"], 5),
                    "sys": round(metrics["sys"]["sen"], 5),
                    "dia": round(metrics["dia"]["sen"], 5)
                }
                for segment in ["s1", "s2", "sys", "dia"]:
                    precision_collection[segment].append(paired_precision[fname][segment])
                    sensitivity_collection[segment].append(paired_sensitivity[fname][segment])
                    
                    
                
                metrics = o['metrics']
                # Add metrics one by one 
                for d in ["s1","s2","sys","dia"]:
                    for k in ["sen", "pre", "f1"]:
                        metrics_[d][k] += metrics[d][k]
            else:
                pass
        
        
                    
        if save:
        # Divide test data
            for d in ["s1","s2","sys","dia"]:
                for k in ["sen", "pre", "f1"]: 
                        metrics_[d][k] /= len(outputs)
                        metrics_[d][k]= round(metrics_[d][k],4)


            plt.rcParams['figure.figsize'] = (9,9)
            species = ['s1', 's2', 'systolic', 'diastole']
            penguin_means = {'Sensitivity': [], 'Precision': [], 'F1_Score': []}

            for h in range(len(keys_)):
                penguin_means['Sensitivity'].append(metrics_[keyss[h]]['sen'])
                penguin_means['Precision'].append(metrics_[keyss[h]]['pre'])
                penguin_means['F1_Score'].append(metrics_[keyss[h]]['f1'])

            
            # Print out penguin_means
            formatted_means = {}
            f1_score_collection = {"s1": [], "s2": [], "sys": [], "dia": []}  
            segment_names = ["s1", "s2", "sys", "dia"]

            for i, segment in enumerate(segment_names): # ['s1', 's2', 'sys', 'dia']
                values = [penguin_means[metric][i] for metric in ['Sensitivity', 'Precision', 'F1_Score']]
                formatted_means[segment] = values

            print("penguin_means:", formatted_means)

            for fname, scores in paired_test.items():
                for segment in segment_names:
                    f1_score_collection[segment].append(scores[segment])

            np.save(f'{self.path}fname_f1_score.npy', paired_test)
            np.save(f'{self.path}f1_score_collection.npy', f1_score_collection)


            # precision_collection, sensitivity_collection 저장하기
            np.save(f'{self.path}fname_pre_score.npy', paired_precision)
            np.save(f'{self.path}pre_score_collection.npy', precision_collection)
            np.save(f'{self.path}fname_sen_score.npy', paired_sensitivity)
            np.save(f'{self.path}sen_score_collection.npy', sensitivity_collection)


            # Save as txt
            with open(f'{self.path}PCG_Metrics_{self.year}_toler{self.toler}_result.txt', 'w') as file:
                file.write("penguin_means:\n")
                for segment in segment_names: # ['s1', 's2', 'sys', 'dia']
                    values = [penguin_means[metric][i] for metric in ['Sensitivity', 'Precision', 'F1_Score']]
                    file.write(f"{segment}: {values}\n")


            x_ = np.arange(len(species))
            width = 0.25  # the width of the bars
            multiplier = 0

            fig, ax = plt.subplots(layout='constrained')

            for attribute, measurement in penguin_means.items():
                offset = width * multiplier
                rects = ax.bar(x_+offset, measurement, width, label=attribute,)
                ax.bar_label(rects, padding=15)
                multiplier += 1

            ax.set_title('PCG_Metrics',fontsize=17)
            ax.set_xticks(x_ + width, species)
            ax.set_xticklabels(species, fontsize=16)
            ax.legend(loc='upper left', ncols=3, fontsize=12.5
                    )
            ax.set_ylim(0, np.max(penguin_means['F1_Score'])+0.3)
            plt.subplots_adjust(top=0.8) # this line will adjust the distance between the title and the graph

            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d_%H_%M")

            plt.savefig(f'{self.path}PCG_Metrics_{self.year}_toler{self.toler}_.png', dpi=300, bbox_inches='tight') 
            

    @staticmethod
    def apply_threshold(pred, t):
        try:
            result = pred.clone()
        except:
            result = pred.copy()
        result[result>=t]= 1
        result[result<t]= 0
        return result
    
    @staticmethod
    def postprocess(single_array):
        single_array = single_array.astype(bool)
        single_array = skimage.morphology.remove_small_objects(single_array, min_size=50, connectivity=1).astype(int)
        return single_array
    
    @staticmethod
    def get_Binaryindex(arr):
        """
        get indexes
        """
        idxs = []
        arr_ = arr.copy()
        arr_ = arr_.round()

        label_result, count = scipy.ndimage.label(arr_)
        for i in range(1,count+1):
            index = np.where(label_result == i)[0]
            start = index[0]
            end = index[-1]
            # print(end-start)
            idxs.append(int(np.mean([start,end])))
        return idxs

