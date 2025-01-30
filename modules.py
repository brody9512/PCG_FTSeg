import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers import same_padding
from monai.networks.layers.factories import Conv

from utils import logsumexp_2d


# class NLBlockND(nn.Module):
#     def __init__(self, in_channels, inter_channels=None, mode='embedded', dimension=3, norm_layer='batch'):
#         super().__init__()
#         assert dimension in [1, 2, 3]
        
#         if mode not in ['gaussian', 'embedded', 'dot', 'concatenate']:
#             raise ValueError('`mode` must be one of `gaussian`, `embedded`, `dot` or `concatenate`')
            
#         self.mode = mode
#         self.dimension = dimension
#         self.in_channels = in_channels
#         self.inter_channels = inter_channels

#         if self.inter_channels is None:
#             self.inter_channels = in_channels // 2
#             if self.inter_channels == 0:
#                 self.inter_channels = 1
        
#         if dimension == 3:
#             conv_nd = nn.Conv3d
#             bn = nn.InstanceNorm3d if norm_layer == 'instance' else nn.BatchNorm3d           
#         elif dimension == 2:
#             conv_nd = nn.Conv2d
#             bn = nn.InstanceNorm3d if norm_layer == 'instance' else nn.BatchNorm3d
#         else:
#             conv_nd = nn.Conv1d
#             bn = nn.InstanceNorm3d if norm_layer == 'instance' else nn.BatchNorm3d

#         self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)

#         if norm_layer is not None:
#             self.W_z = nn.Sequential(
#                     conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1),
#                     bn(self.in_channels)
#                 )
#         else:
#             self.W_z = conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels, kernel_size=1)

#             nn.init.constant_(self.W_z.weight, 0)
#             nn.init.constant_(self.W_z.bias, 0)

#         if self.mode in ['embedded', 'dot', 'concatenate']:
#             self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
#             self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        
#         if self.mode == "concatenate":
#             self.W_f = nn.Sequential(
#                     conv_nd(self.inter_channels * 2, 1, kernel_size=1),
#                     nn.LeakyReLU(0.1)
#                 )
            
#     def forward(self, x):
#         """
#         args
#             x: (N, C, T, H, W) for dimension=3; (N, C, H, W) for dimension 2; (N, C, T) for dimension 1
#         """

#         batch_size = x.size(0)
        
#         g_x = self.g(x).view(batch_size, self.inter_channels, -1)
#         g_x = g_x.permute(0, 2, 1)

#         if self.mode == "gaussian":
#             theta_x = x.view(batch_size, self.in_channels, -1).permute(0, 2, 1)
#             phi_x = x.view(batch_size, self.in_channels, -1)
#             f = torch.matmul(theta_x, phi_x)

#         elif self.mode == "embedded" or self.mode == "dot":
#             theta_x = self.theta(x).view(batch_size, self.inter_channels, -1).permute(0, 2, 1)
#             phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
#             f = torch.matmul(theta_x, phi_x)

#         else:
#             theta_x = self.theta(x).view(batch_size, self.inter_channels, -1, 1)
#             phi_x = self.phi(x).view(batch_size, self.inter_channels, 1, -1)
#             h, w = theta_x.size(2), phi_x.size(3)
#             theta_x = theta_x.repeat(1, 1, 1, w)
#             phi_x = phi_x.repeat(1, 1, h, 1)
#             concat = torch.cat([theta_x, phi_x], dim=1)
#             f = self.W_f(concat)
#             f = f.view(f.size(0), f.size(2), f.size(3))
        
#         if self.mode == "gaussian" or self.mode == "embedded":
#             f_div_C = F.softmax(f, dim=-1)
            
#         elif self.mode in ['dot', 'concatenate']:
#             N = f.size(-1)
#             f_div_C = f / float(N)
        
#         y = torch.matmul(f_div_C, g_x).permute(0, 2, 1).contiguous()
#         y = y.view(batch_size, self.inter_channels, *x.size()[2:])
#         W_y = self.W_z(y)
#         z = W_y + x

#         return z
    
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

# class Flatten(nn.Module):
#     def forward(self, x):
#         return x.view(x.size(0), -1)

# class CBAM(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=('avg', 'max'), no_spatial=False):
#         super().__init__()
#         self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
#         self.no_spatial=no_spatial
#         if not no_spatial:
#             self.SpatialGate = SpatialGate()
#     def forward(self, x):
#         out = self.ChannelGate(x)
#         if not self.no_spatial:
#             out = self.SpatialGate(out)
#         return out
# class ChannelGate(nn.Module):
#     def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
#         super().__init__()
#         self.gate_channels = gate_channels
#         self.mlp = nn.Sequential(
#             Flatten(),
#             nn.Linear(gate_channels, gate_channels // reduction_ratio),
#             nn.ReLU(),
#             nn.Linear(gate_channels // reduction_ratio, gate_channels)
#             )
#         self.pool_types = pool_types
        
#     def forward(self, x):
#         channel_att_sum = None
#         for pool_type in self.pool_types:
#             if pool_type=='avg':
#                 pool = F.avg_pool1d( x, x.size(2), stride=x.size(2))
#             elif pool_type=='max':
#                 pool = F.max_pool1d( x, x.size(2), stride=x.size(2))
#             elif pool_type=='lp':
#                 pool = F.lp_pool1d( x, 2, x.size(2), stride=x.size(2))
#             elif pool_type=='lse':
#                 pool = logsumexp_2d(x)

#             channel_att_raw = self.mlp(pool)

#             if channel_att_sum is None:
#                 channel_att_sum = channel_att_raw
#             else:
#                 channel_att_sum += channel_att_raw

#         scale = torch.sigmoid(channel_att_sum).unsqueeze(2).expand_as(x)
#         return x * scale
# class ChannelPool(nn.Module):
#     def forward(self, x):
#         return torch.cat((
#             torch.max(x, dim=1, keepdim=True)[0],
#             torch.mean(x, dim=1, keepdim=True)
#         ), dim=1)
# class SpatialGate(nn.Module):
#     def __init__(self, kernel_size=7):
#         super().__init__()
#         self.compress = ChannelPool()
#         padding = (kernel_size - 1) // 2
#         self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=padding, relu=False)
#     def forward(self, x):
#         x_compress = self.compress(x)
#         x_out = self.spatial(x_compress)
#         scale = torch.sigmoid(x_out)
#         return x * scale

# class SELayer1D(nn.Module):
#     def __init__(self, channel, reduction=args.se_ratio, acti_type_1='LeakyReLU', acti_type_2='ReLU'):
#         super(SELayer1D, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(channel, max(1, channel // reduction), bias=False),
#             getattr(nn, acti_type_1)(inplace=True) if acti_type_1 != 'Sigmoid' else getattr(nn, acti_type_1)(),
#             nn.Linear(max(1, channel // reduction), channel, bias=False),
#             getattr(nn, acti_type_2)()
#         )

#     def forward(self, x):
#         b, c, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1)
#         return x * y.expand_as(x)
# class SEBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=args.se_ratio, acti_type_1='LeakyReLU', acti_type_2='ReLU'):
#         super(SEBlock1D, self).__init__()
#         # Ensure you're using the correct case for activation functions
#         self.relu1 = getattr(nn, acti_type_1)(inplace=True) if acti_type_1 != 'Sigmoid' else getattr(nn, acti_type_1)()
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)  # BatchNorm1d 초기화 추가
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.se = SELayer1D(out_channels, reduction=reduction, acti_type_1=acti_type_1, acti_type_2=acti_type_2)

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = self.se(out)
#         out = self.relu1(out)  # Using relu1 for simplicity. Adjust as needed.
#         return out
# class ResidualSEBlock1D(nn.Module):
#     def __init__(self, in_channels, out_channels, reduction=args.se_ratio, acti_type_1='LeakyReLU', acti_type_2='ReLU'):
#         super(ResidualSEBlock1D, self).__init__()  # Fixed incorrect superclass reference
#         self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm1d(out_channels)
#         self.relu1 = getattr(nn, acti_type_1)(inplace=True) if acti_type_1 != 'Sigmoid' else getattr(nn, acti_type_1)()
#         self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm1d(out_channels)
#         self.se = SELayer1D(out_channels, reduction=reduction, acti_type_1=acti_type_1, acti_type_2=acti_type_2)

#         self.shortcut = nn.Sequential(
#             nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
#             nn.BatchNorm1d(out_channels)
#         ) if in_channels != out_channels else nn.Identity()

#     def forward(self, x):
#         out = self.relu1(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out = self.se(out)
#         shortcut = self.shortcut(x)
#         out += shortcut
#         out = self.relu1(out)  # Using relu1 for simplicity. Adjust as needed.
#         return out
# class DeepRFT_SE_identity(nn.Module):
#     def __init__(self, in_channels, out_channels, norm='backward',not_se=False,not_fft=False, residual_one=False, img_not_residual_one=False, spatial_dims=1):
#         super(DeepRFT_SE_identity, self).__init__()
#         ## kernel_size=1,3 각각 다른 이유??
#         self.img_conv  = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#         self.fft_conv  = nn.Sequential(nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
#                                        nn.Conv1d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0))
#         self.norm1 = nn.InstanceNorm1d(out_channels)
#         self.norm2 = nn.InstanceNorm1d(out_channels*2)
        
#         self.resi_seblock =ResidualSEBlock1D(in_channels, out_channels, reduction=args.se_ratio, acti_type_1='LeakyReLU', acti_type_2='ReLU')  #monai.networks.blocks.ResidualSELayer(spatial_dims=spatial_dims, in_channels=in_channels, r=args.se_ratio, acti_type_1='leakyrelu', acti_type_2='relu')
#         self.seblock= SEBlock1D(in_channels, out_channels, reduction=args.se_ratio, acti_type_1='LeakyReLU', acti_type_2='ReLU')
        
#         self.residual_one = residual_one
#         self.img_not_residual_one=img_not_residual_one
#         self.not_se=not_se
#         self.not_fft=not_fft
        
#     def forward(self, x):
#         # Fourier domain   
#         # _, _, W = x.shape
#         fft = torch.fft.rfft(x, norm='ortho')
#         fft = torch.cat([fft.real, fft.imag], dim=1) 

#         # fft = F.leaky_relu(self.norm2(self.fft_conv(fft)),0.01)
#         fft = F.gelu(self.norm2(self.fft_conv(fft)))
        
#         fft_real, fft_imag = torch.chunk(fft, 2, dim=1) 
#         ##chunk함수는 tensor를 쪼개는 함수이다. tensor를 몇개로 어떤 dimension으로 쪼갤지 설정해주고 사용        
        
#         fft = torch.complex(fft_real, fft_imag)
#         '''real = torch.tensor([1, 2], dtype=torch.float32)
#         imag = torch.tensor([3, 4], dtype=torch.float32)
#         z = torch.complex(real, imag)
#         z
#         tensor([(1.+3.j), (2.+4.j)])'''
        
#         fft = torch.fft.irfft(fft, norm='ortho')
#         # fft = self.norm1(fft)
#         # Image domain  
        
#         # img = F.leaky_relu(self.norm1(self.img_conv(x)),0.01)
#         img = F.gelu(self.norm1(self.img_conv(x)))

#         if self.residual_one:
#             if self.not_se:
#                 output = x + img + fft
#             if self.not_fft:
#                 output = x + img + self.seblock(x)
#             if self.not_se and self.not_fft:
#                 output = x + img
                
#             output = x + img + fft + self.seblock(x)    #+ self.seblock(x)
#         else:
#             if self.img_not_residual_one:
#                  output = x + fft + self.seblock(x)
            
#             else:    # Mixing (residual, image, fourier)
#                 output = x + img + fft + self.resi_seblock(x)
#         return output


# class convRFT(nn.Module):
#     def __init__(self, in_channels, out_channels, norm='backward'):
#         super(convRFT, self).__init__()
#         ## kernel_size=1,3 각각 다른 이유??
#         self.img_conv  = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#         self.fft_conv  = nn.Sequential(nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
#                                        nn.Conv1d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0))
#         self.norm1 = nn.InstanceNorm1d(out_channels)
#         self.norm2 = nn.InstanceNorm1d(out_channels*2)

#     def forward(self, x):
#         # Fourier domain   
#         fft = torch.fft.rfft(x, norm='ortho')
#         fft = torch.cat([fft.real, fft.imag], dim=1) 

#         fft = F.gelu(self.norm2(self.fft_conv(fft)))
        
#         fft_real, fft_imag = torch.chunk(fft, 2, dim=1) 
#         ##chunk함수는 tensor를 쪼개는 함수이다. tensor를 몇개로 어떤 dimension으로 쪼갤지 설정해주고 사용        
        
#         fft = torch.complex(fft_real, fft_imag)
#         '''real = torch.tensor([1, 2], dtype=torch.float32)
#         imag = torch.tensor([3, 4], dtype=torch.float32)
#         z = torch.complex(real, imag)
#         z
#         tensor([(1.+3.j), (2.+4.j)])'''
        
#         fft = torch.fft.irfft(fft, norm='ortho')
#         # Image domain  
        
#         img = F.gelu(self.norm1(self.img_conv(x)))

#         # Mixing (residual, image, fourier)
#         output = img
#         return output
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
# class fftconvRFT(nn.Module):
#     def __init__(self, in_channels, out_channels, norm='backward'):
#         super(fftconvRFT, self).__init__()
#         ## kernel_size=1,3 각각 다른 이유??
#         self.img_conv  = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#         self.fft_conv  = nn.Sequential(nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
#                                        nn.Conv1d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0))
#         self.norm1 = nn.InstanceNorm1d(out_channels)
#         self.norm2 = nn.InstanceNorm1d(out_channels*2)

#     def forward(self, x):
#         # Fourier domain   
#         # _, _, W = x.shape
#         fft = torch.fft.rfft(x, norm='ortho')
#         fft = torch.cat([fft.real, fft.imag], dim=1) 

#         # fft = F.leaky_relu(self.norm2(self.fft_conv(fft)),0.01)
#         fft = F.gelu(self.norm2(self.fft_conv(fft)))
        
#         fft_real, fft_imag = torch.chunk(fft, 2, dim=1) 
#         ##chunk함수는 tensor를 쪼개는 함수이다. tensor를 몇개로 어떤 dimension으로 쪼갤지 설정해주고 사용        
        
#         fft = torch.complex(fft_real, fft_imag)
#         '''real = torch.tensor([1, 2], dtype=torch.float32)
#         imag = torch.tensor([3, 4], dtype=torch.float32)
#         z = torch.complex(real, imag)
#         z
#         tensor([(1.+3.j), (2.+4.j)])'''
        
#         fft = torch.fft.irfft(fft, norm='ortho')
#         # fft = self.norm1(fft)
#         # Image domain  
        
#         # img = F.leaky_relu(self.norm1(self.img_conv(x)),0.01)
#         img = F.gelu(self.norm1(self.img_conv(x)))

#         # Mixing (residual, image, fourier)
#         output = img + fft
#         return output

# class DeepRFT(nn.Module):
#     def __init__(self, in_channels, out_channels, norm='backward'):
#         super(DeepRFT, self).__init__()
#         ## kernel_size=1,3 각각 다른 이유??
#         self.img_conv  = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
#                                        nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
#         self.fft_conv  = nn.Sequential(nn.Conv1d(in_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0),
#                                        nn.Conv1d(out_channels*2, out_channels*2, kernel_size=1, stride=1, padding=0))
#         self.norm1 = nn.InstanceNorm1d(out_channels)
#         self.norm2 = nn.InstanceNorm1d(out_channels*2)

#     def forward(self, x):
#         # Fourier domain   
#         # _, _, W = x.shape
#         fft = torch.fft.rfft(x, norm='ortho')
#         fft = torch.cat([fft.real, fft.imag], dim=1) 

#         # fft = F.leaky_relu(self.norm2(self.fft_conv(fft)),0.01)
#         fft = F.gelu(self.norm2(self.fft_conv(fft)))
        
#         fft_real, fft_imag = torch.chunk(fft, 2, dim=1) 
#         ##chunk함수는 tensor를 쪼개는 함수이다. tensor를 몇개로 어떤 dimension으로 쪼갤지 설정해주고 사용        
        
#         fft = torch.complex(fft_real, fft_imag)
#         '''real = torch.tensor([1, 2], dtype=torch.float32)
#         imag = torch.tensor([3, 4], dtype=torch.float32)
#         z = torch.complex(real, imag)
#         z
#         tensor([(1.+3.j), (2.+4.j)])'''
        
#         fft = torch.fft.irfft(fft, norm='ortho')
#         # fft = self.norm1(fft)
#         # Image domain  
        
#         # img = F.leaky_relu(self.norm1(self.img_conv(x)),0.01)
#         img = F.gelu(self.norm1(self.img_conv(x)))

#         # Mixing (residual, image, fourier)
#         output = x + img + fft
#         return output


# class SimpleASPP(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         conv_out_channels: int,
#         kernel_sizes: Sequence[int] = (1, 3, 3, 3),
#         dilations: Sequence[int] = (1, 2, 4, 6),
#         norm_type: Optional[Union[Tuple, str]] = "BATCH",
#         acti_type: Optional[Union[Tuple, str]] = "LEAKYRELU",
#         bias: bool = False,
#     ) -> None:

#         super().__init__()
#         if len(kernel_sizes) != len(dilations):
#             raise ValueError(
#                 "kernel_sizes and dilations length must match, "
#                 f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
#             )
#         pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

#         self.convs = nn.ModuleList()
#         for k, d, p in zip(kernel_sizes, dilations, pads):
#             _conv = Conv[Conv.CONV, spatial_dims](
#                 in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
#             )

#             self.convs.append(nn.Sequential(_conv,
#                                              #DeepRFT(conv_out_channels, conv_out_channels)
#                                              ))
            
#         out_channels = conv_out_channels * len(pads) 
#         self.conv_k1 = Convolution(
#             spatial_dims=spatial_dims,
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             act=acti_type,
#             norm=norm_type,
#             bias=bias,
#         )


#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
#         """
#         # x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
#         convs = list()
#         for conv in self.convs:
#             convs.append(conv(x))
#         x_out = torch.cat(convs, dim=1)
#         x_out = self.conv_k1(x_out)
#         return x_out

# class SimpleASPP_deeprft(nn.Module):
#     def __init__(
#         self,
#         spatial_dims: int,
#         in_channels: int,
#         conv_out_channels: int,
#         kernel_sizes: Sequence[int] = (1, 3, 3, 3),
#         dilations: Sequence[int] = (1, 2, 4, 6),
#         norm_type: Optional[Union[Tuple, str]] = "BATCH",
#         acti_type: Optional[Union[Tuple, str]] = "LEAKYRELU",
#         bias: bool = False,
#     ) -> None:

#         super().__init__()
#         if len(kernel_sizes) != len(dilations):
#             raise ValueError(
#                 "kernel_sizes and dilations length must match, "
#                 f"got kernel_sizes={len(kernel_sizes)} dilations={len(dilations)}."
#             )
#         pads = tuple(same_padding(k, d) for k, d in zip(kernel_sizes, dilations))

#         self.convs = nn.ModuleList()
#         for k, d, p in zip(kernel_sizes, dilations, pads):
#             _conv = Conv[Conv.CONV, spatial_dims](
#                 in_channels=in_channels, out_channels=conv_out_channels, kernel_size=k, dilation=d, padding=p
#             )

#             self.convs.append(nn.Sequential(_conv,
#                                              DeepRFT(conv_out_channels, conv_out_channels)
#                                              ))
            
#         out_channels = conv_out_channels * len(pads)  
#         self.conv_k1 = Convolution(
#             spatial_dims=spatial_dims,
#             in_channels=out_channels,
#             out_channels=out_channels,
#             kernel_size=1,
#             act=acti_type,
#             norm=norm_type,
#             bias=bias,
#         )

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """
#         Args:
#             x: in shape (batch, channel, spatial_1[, spatial_2, ...]).
#         """
#         # x_out = torch.cat([conv(x) for conv in self.convs], dim=1)
#         convs = list()
#         for conv in self.convs:
#             convs.append(conv(x))
#         x_out = torch.cat(convs, dim=1)
#         x_out = self.conv_k1(x_out)
#         return x_out
