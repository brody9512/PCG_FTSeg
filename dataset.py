import monai
import numpy as np
import torch 

# Import from Directory Architecture
from utils import augment_neurokit, zscore

class PCGDataset():
    """
    A PyTorch Dataset class for loading and processing PCG data.
    """
    def __init__(self, args, data, phase='test'):
        self.data = data
        self.phase = phase
        self.target_sr = args.target_sr
        self.featureLength = args.featureLength
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        # Extract raw audio and segmentation
        audio = self.data[idx]['wav']
        seg = self.data[idx]['seg']
        fname = self.data[idx]['fname']
        fname =self.data[idx]['fname']

        # Remove leading/trailing zeros in the segmentation & match audio length
        valid_indices = np.where(seg != 0)[0]
        start_idx, end_idx = valid_indices[0], valid_indices[-1]
        audio = audio[start_idx:end_idx]
        seg = seg[start_idx:end_idx]

        # Data augmentation during training
        if self.phase == 'train':
            audio = augment_neurokit(audio, self.target_sr)

            total_len = audio.shape[-1]
            delta = (total_len - self.featureLength) // 2 #2016은 featureLength=8192
            random_shift = int(np.random.rand() * delta)
            start = random_shift
            end = start + self.featureLength

            audio_segment = audio[start:end]
            seg_segment = seg[start:end]

            # Random amplitude scaling
            p = 0.2
            if np.random.rand() <= p:
                scaling_factor = (np.random.rand() - 0.5) * 2
                audio_segment = audio_segment * scaling_factor
        else:
            audio_segment = audio
            seg_segment = seg

        # Convert to tensors
        audio_tensor = torch.from_numpy(audio_segment).unsqueeze(0).float() # is to float() needed??
        audio_tensor = zscore(audio_tensor)
        
        # Create a 2-channel input:
        #   1) the raw (z-scored) waveform
        #   2) absolute magnitude (amplitude envelope)
        audio_2ch = torch.cat([audio_tensor, torch.sqrt(audio_tensor**2)], dim=0) # x_ = torch.concat([x_, torch.sqrt(x_**2)],dim=0) # original and amplitude as input

        # Convert segmentation to tensor
        try:
            seg_tensor_4class = torch.from_numpy(seg_segment)
        except ValueError:
            seg_tensor_4class = torch.zeros(audio_2ch.shape[-1])
        seg_tensor_4class = seg_tensor_4class.unsqueeze(0).long() ## is to long() needed??

        # Re-map the segmentation values (just clarifying steps)
        # Possible classes: {1,2,3,4} or {0,...}
        seg_tensor_4class[seg_tensor_4class == 2] = 2
        seg_tensor_4class[seg_tensor_4class == 4] = 4
        seg_tensor_4class[seg_tensor_4class == 1] = 1
        seg_tensor_4class[seg_tensor_4class == 3] = 3

        # Create 2-class segmentation clone
        seg_tensor_2class = seg_tensor_4class.clone()
        # Map {2->1,4->1,1->0,3->0}
        seg_tensor_2class[seg_tensor_4class == 2] = 1
        seg_tensor_2class[seg_tensor_4class == 4] = 1
        seg_tensor_2class[seg_tensor_4class == 1] = 0
        seg_tensor_2class[seg_tensor_4class == 3] = 0

        # Build one-hot (5 classes => final slice is used for 2-class separation)
        seg_fourclass_zero_based = seg_tensor_4class - 1
        seg_one_hot = monai.networks.utils.one_hot(seg_fourclass_zero_based, num_classes=5, dim=0)
        # Replace last channel with 2-class mask
        seg_one_hot[-1] = seg_tensor_2class[0]

        # Swap the first row with the last row, then keep [1:]
        seg_one_hot[0], seg_one_hot[-1] = seg_one_hot[-1].clone(), seg_one_hot[0].clone()
        seg_one_hot = seg_one_hot[1:]


        #####
        try:
            y_ = torch.from_numpy(y_)
        except:
            y_ = torch.zeros(x_.shape[-1])
        y_ = torch.unsqueeze(y_,0)
        
        x_=x_.float()
        y_=y_.long()

        y_[y_==2]=2
        y_[y_==4]=4
        y_[y_==1]=1
        y_[y_==3]=3 # 4ch
        y__ = y_.clone()
        # y__[y__!=0]=1

        y__[y_==2]=1
        y__[y_==4]=1
        y__[y_==1]=0
        y__[y_==3]=0 # 2ch

        y_onehot = y_.clone()

        # Subtract 1 from the tensor's values to make them zero-based
        y_onehot1_adjusted = y_onehot - 1
        
        # Call one_hot function with the correct number of classes (5) and the adjusted tensor
        y_onehot1 = monai.networks.utils.one_hot(y_onehot1_adjusted, 5, dim=0)
        y_onehot1[-1]=y__[0]
        
        # Swap the first row (y0[0]) with the last row (y0[-1])
        y_onehot1[0], y_onehot1[-1] = y_onehot1[-1].clone(), y_onehot1[0].clone()
        y_onehot1=y_onehot1[1:]
        # y_onehot = y_onehot[1:]
        # y_onehot = torch.flip(y_onehot,dims=(0,))
        return {'x':x_, 'y':y__, 'y_4stage':y_, 'y_onehot': y_onehot1, 'fname':fname}
        # return {'x':x_, 'y':y__, 'y_2stage':y_, 'y_onehot': y_onehot, 'fname':fname}
  