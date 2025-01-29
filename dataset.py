import monai
import numpy as np
import torch 

# Import from Directory Architecture
from config import get_args
from utils import augment_neurokit, zscore

args = get_args()

class dataset():
    def __init__(self, data, phase='test'):
        self.data = data
        self.phase = phase
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        
        x = self.data[idx]['wav']
        y = self.data[idx]['seg']
        fname =self.data[idx]['fname']
        # x = butter_bandpass_filter(x,target_sr,20,200) #이거 맞춰서 20-200가야 하는 건가

        # seg가 0인 것을 확실히 제거한다, 이 참에 wav도 똑같이 한다
        no_zeros=np.where(y != 0)[0]
        x = x[no_zeros[0]:no_zeros[-1]]
        y = y[no_zeros[0]:no_zeros[-1]]        
        
        if self.phase=='train':
            x = augment_neurokit(x, args.target_sr)
            total = x.shape[-1]
            delta = (total - args.featureLength)//2 #2016은 featureLength=8192
            rn = int(np.random.rand()* delta) 
            start = rn
            end = start+args.featureLength
            x_=x[start:end]
            y_=y[start:end]
            
            p = 0.2
            rn = np.random.rand(1)
            if rn <= p:
                x = x*(rn-.5)*2
        else:
            x_ = x
            y_ = y
        
        x_ = torch.from_numpy(x_)
        x_ = torch.unsqueeze(x_,0)
        #i = torch.rand(3, 20, 128)
        #i = i.unsqueeze(dim=1) #[3, 20, 128] -> [3, 1, 20, 128]
        x_ = zscore(x_)
        # x_ = minmax(x_)
        x_ = torch.concat([x_, torch.sqrt(x_**2)],dim=0) # original and amplitude as input
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
  