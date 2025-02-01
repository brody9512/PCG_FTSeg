import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
import numpy as np
import neurokit2 as nk


def set_seed(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    monai.utils.misc.set_determinism(seed=seed)

class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = monai.losses.DiceLoss()
   
    def forward(self, inputs, targets):
        dice_loss  = self.dice(inputs, targets)
        bce_loss  = F.binary_cross_entropy(inputs, targets)
        return dice_loss + bce_loss
    
def eval_metrics(gt_idxs, pred_idxs, tp_accum, fn_accum, fp_accum, sen_accum, pre_accum, f1_accum, tolerance=40):
    """
    Evaluate event-based segmentation metrics with a tolerance window.
    gt_idxs: list of ground-truth event indices
    pred_idxs: list of predicted event indices
    """
    TP = 0
    # Count true positives with tolerance
    for g in gt_idxs:
        for p in pred_idxs:
            if abs(g - p) <= tolerance:
                TP += 1
                break

    # Count false positives
    FP = len(pred_idxs) - TP

    # Another pass for FN
    TP2 = 0
    for p in pred_idxs:
        for g in gt_idxs:
            if abs(p - g) <= tolerance:
                TP2 += 1
                break

    FN = len(gt_idxs) - TP2

    if (TP + FN) == 0:
        sensitivity = 0
    else:
        sensitivity = TP / (TP + FN)

    if (TP + FP) == 0:
        precision = 0
    else:
        precision = TP / (TP + FP)

    if (sensitivity + precision) == 0:
        f1_score = 0
    else:
        f1_score = 2 * sensitivity * precision / (sensitivity + precision)

    # Accumulate into running totals
    tp_accum += TP
    fn_accum += FN
    fp_accum += FP
    sen_accum += sensitivity
    pre_accum += precision
    f1_accum += f1_score

    return tp_accum, fn_accum, fp_accum, sen_accum, pre_accum, f1_accum
    
def zscore(arr,mean=None,std=None):
    if mean!=None or std!=mean:
        return (arr-mean)/(std+1e-8)
    else:
        try:
            return (arr-np.mean(arr))/(np.std(arr)+1e-8)
        except:
            return (arr-torch.mean(arr))/(torch.std(arr)+1e-8)

def augment_neurokit(pcg_signal, sr, p=0.2):
    """
    Randomly distorts a PCG signal using NeuroKit's signal_distort.
    """
    if np.random.rand(1) <= p:
        noise_shape = np.random.choice(['gaussian', 'laplace'])
        noise_amplitude = np.random.rand(1)*.4 #/ noise_frequency
        powerline_amplitude = np.random.rand(1)*.2 #/ powerline_frequency
        artifacts_amplitude = np.random.rand(1)*1 #/ artifacts_frequency
        
        noise_frequency = np.random.randint(10,50)
        powerline_frequency = np.random.randint(50,60)
        artifacts_frequency= np.random.randint(2,40)
        artifacts_number = 10

        pcg_signal = nk.signal_distort(
            pcg_signal,
            sampling_rate=sr,
            noise_shape=noise_shape,
            noise_amplitude=noise_amplitude,
            powerline_amplitude=powerline_amplitude,
            artifacts_amplitude=artifacts_amplitude,
            noise_frequency=noise_frequency,
            powerline_frequency=powerline_frequency,
            artifacts_frequency=artifacts_frequency,
            artifacts_number=artifacts_number,
            linear_drift=False,
            random_state=42,
            silent=True
        )
    return pcg_signal