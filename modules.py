import torch
import torch.nn as nn
import torch.nn.functional as F

    
class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                 relu=True, bn=True, bias=False):
        super().__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm1d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class fftRFT(nn.Module):
    def __init__(self, in_channels, out_channels, norm='backward'):
        super(fftRFT, self).__init__()
        self.img_conv  = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                                       nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        self.fft_conv  = nn.Sequential(nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
                                       nn.Conv1d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0))
        self.norm1 = nn.InstanceNorm1d(out_channels)
        self.norm2 = nn.InstanceNorm1d(out_channels*2)

    def forward(self, x):
        # Fourier domain   
        # _, _, W = x.shape
        fft = torch.fft.rfft(x, norm='ortho')
        fft = torch.cat([fft.real, fft.imag], dim=1) 

        fft = F.gelu(self.norm2(self.fft_conv(fft)))
        
        fft_real, fft_imag = torch.chunk(fft, 2, dim=1) 
        ##chunk함수는 tensor를 쪼개는 함수이다. tensor를 몇개로 어떤 dimension으로 쪼갤지 설정해주고 사용        
        
        fft = torch.complex(fft_real, fft_imag)
        '''real = torch.tensor([1, 2], dtype=torch.float32)
        imag = torch.tensor([3, 4], dtype=torch.float32)
        z = torch.complex(real, imag)
        z
        tensor([(1.+3.j), (2.+4.j)])'''
        
        fft = torch.fft.irfft(fft, norm='ortho')
        # fft = self.norm1(fft)
        # Image domain  
        
        # img = F.leaky_relu(self.norm1(self.img_conv(x)),0.01)
        img = F.gelu(self.norm1(self.img_conv(x)))

        # Mixing (residual, image, fourier)
        output = fft
        return output