import math

import torch
import torch.nn.functional as F
# import torch.utils.model_zoo as model_zoo
from torch import nn
# from torch.nn import Parameter
# import pdb
# import numpy as np



class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__() 
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            # [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0, dilation=self.conv.dilation, groups=self.conv.groups)

            return out_normal - self.theta * out_diff

        
class SpatialAttention(nn.Module):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)

class HiFiNet(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7 ):   
        super(HiFiNet, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 80, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(80),
            nn.ReLU(),
        )
        
        self.Block1 = nn.Sequential(
            basic_conv(80, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            
            basic_conv(160, int(160*1.6), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.6)),
            nn.ReLU(),
            basic_conv(int(160*1.6), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
        )
        
        self.Block2 = nn.Sequential(
            basic_conv(160,160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv( 160, int(160*1.4),kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.4)),
            nn.ReLU(),
            basic_conv(int(160*1.4), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Cat1 = nn.Sequential(
            basic_conv(160 * 2, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        
        self.Block3 = nn.Sequential(
            basic_conv(160,160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block4 = nn.Sequential(
            basic_conv(160,160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            basic_conv(160, int(160*1.2), kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(int(160*1.2)),
            nn.ReLU(),
            basic_conv(int(160*1.2), 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),

            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block5 = nn.Sequential(
            basic_conv(160*2, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        self.Block6 = nn.Sequential(
            basic_conv(160, 160, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
        )
        # Original
        
        self.lastconv1 = nn.Sequential(
            basic_conv(160*5, 160, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.BatchNorm2d(160),
            nn.ReLU(),
            basic_conv(160, 1, kernel_size=3, stride=1, padding=1, bias=False, theta= theta),
            nn.ReLU(),    
        )

        self.sa1 = SpatialAttention(kernel = 9)
        self.sa2 = SpatialAttention(kernel = 7)
        self.sa3 = SpatialAttention(kernel = 5)
        self.sa4 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32,32), mode='bilinear')
        self.downsample16x16 = nn.Upsample(size=(16,16), mode='bilinear')

 
    def forward(self, x):	    	# x [3, 256, 256]
        
        x_input = x
        x = self.conv1(x)		   
        
        x_Block1 = self.Block1(x)	    	    	
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_16x16 = self.downsample16x16(x_Block1_SA)
        
        x_Block2 = self.Block2(x_Block1)
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_16x16 = self.downsample16x16(x_Block2_SA)
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)

        x_Block3 = self.Block3(x_Block2)
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_16x16 = self.downsample16x16(x_Block3_SA)

        x_cat1 = torch.cat((x_Block2_32x32, x_Block3), dim=1)
        x_Block23 = self.Cat1(x_cat1)

        x_Block4 = self.Block4(x_Block23)
        attention4 = self.sa4(x_Block4)
        x_Block4_SA = attention4 * x_Block4
        x_Block4_16x16 = self.downsample16x16(x_Block4_SA)


        x_Block_res1  = self.Block6(x_Block4)
        x_concat_res = torch.cat((x_Block4 , x_Block_res1 ), dim=1)
        x_Block_res2 = self.Block5(x_concat_res)
        x_concat_res = torch.cat((x_Block_res1 , x_Block_res2), dim=1)
        x_Block_res3 = self.Block5(x_concat_res)
        x_concat_res = torch.cat((x_Block_res2 , x_Block_res3), dim=1)
        x_Block_res4 = self.Block5(x_concat_res)
        x_concat_res = torch.cat((x_Block_res3 , x_Block_res4), dim=1)
        x_Block_res5 = self.Block5(x_concat_res)
        x_concat_res = torch.cat((x_Block_res4 , x_Block_res5), dim=1)
        x_Block_res6 = self.Block5(x_concat_res)

        x_concat = torch.cat((x_Block1_16x16,x_Block2_16x16,x_Block3_16x16,x_Block4_16x16,x_Block_res6), dim=1)

        map_x = self.lastconv1(x_concat)
        
        map_x = map_x.squeeze(1)

        return map_x



