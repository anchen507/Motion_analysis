import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import PixelUnshuffle
import numpy as np
from M_PWCNet.correlation_package.correlation import Correlation
from utils.warp_utils import flow_warp
class SDNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, nf=64):
        super(SDNet, self).__init__()
        self.in_channel = in_channel
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.spatial_feature = spatial_denoising(filter_in=64,filter_out=64,groups=1)


    def forward(self, x):

        N,C,H,W = x.size()

        residual=x

        out = self.input(x)

        out1 = self.spatial_feature(out,[N,C,H,W])

        out = torch.add(out1, residual)

        return out
        
class TRNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=3, nf=64):
        super(TRNet, self).__init__()
        self.in_channel = in_channel
        self.input = nn.Sequential(
            nn.Conv3d(in_channels=in_channel, out_channels=nf, kernel_size=(1,3,3), stride=1, padding=(0,1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )
        self.spatiotemporal_feature = spatiotemporal_denoising(filter_in=64,filter_out=64,groups=1)

    def forward(self, x):

        N,C,L,H,W = x.size()


        residual=x[:,:,int(L//2):int(L//2)+1,:,:]

        out = self.input(x)

        out1 = self.spatiotemporal_feature(out,[N,C,L,H,W])

        out = torch.add(out1, residual)

        return out

class FRNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=1, nf=64):
        super(FRNet, self).__init__()
        self.in_channel = in_channel
        self.search_range = 4
        
        self.input = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.conv1X1 = nn.Sequential(
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(1,1), stride=1, padding=(0,0), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(1,1), stride=1, padding=(0,0), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.flow_refine= nn.Sequential(
            nn.Conv2d(in_channels=81+nf+2, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=2, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)
        )

        self.image_refine = nn.Sequential(
            nn.Conv2d(in_channels=in_channel*3, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=(3,3), stride=1, padding=(1,1), bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=nf, out_channels=in_channel, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)
            #nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

        self.corr = Correlation(pad_size=self.search_range, kernel_size=1,
                                max_displacement=self.search_range, stride1=1,
                                stride2=1, corr_multiply=1)

        self.LeakyReLU=nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x1,x2,x3):

        N,C,H,W = x1.size()
        residual=x1

        x1_feature=self.input(x1)
        x2_feature=self.input(x2)

        #c_feature= torch.cat((x1_feature,x2_feature),1)

        correlation=self.LeakyReLU(self.corr(x1_feature,x2_feature))

        x2_res = self.conv1X1(x2_feature)
        
        flow_res = self.flow_refine(torch.cat((correlation,x2_res,x3),1))
        
        flow=x3+flow_res
        
        warp_x1=flow_warp(x1,flow)
        
        img_refine=self.image_refine(torch.cat((x1,warp_x1,x2),1))
        
        return img_refine 



class down_sample_spatial(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(down_sample_spatial, self).__init__()
        self.conv = nn.Conv2d(filter_in*4, filter_out , (1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,H,W=shape
        out = PixelUnshuffle.pixel_unshuffle(input,2)
        out = self.lrule(self.conv(out))

        return out


class up_sample_spatial(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(up_sample_spatial, self).__init__()
        self.conv = nn.Conv2d(filter_in, filter_out*4 , (1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,H,W=shape
        out = self.lrule(self.conv(input))
        out = F.pixel_shuffle(out,2)
        return out


class Seq_conv(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,concat_filter=64, groups=1):
        super(Seq_conv, self).__init__()
        self.conv1 = nn.Conv2d(concat_filter, filter_out , (1,1), 1, (0,0), groups=groups, bias=True)
        self.conv2 = nn.Conv2d(filter_in, filter_out , (3,3), 1, (1,1), groups=groups, bias=False)
        self.conv3 = nn.Conv2d(filter_in, filter_out , (3,3), 1, (1,1), groups=groups, bias=False)

        self.BN2 = nn.BatchNorm2d(filter_in, affine=True)
        self.BN3 = nn.BatchNorm2d(filter_in, affine=True)

        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        residual=self.conv1(input)
        out=self.lrule(self.BN2(self.conv2(residual)))
        out=self.lrule(self.BN3(self.conv3(out)))
        return out+residual


class Seq_conv_tail(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(Seq_conv_tail, self).__init__()
        self.conv = nn.Conv2d(filter_in, filter_out , (3,3), 1, (1,1), groups=groups, bias=False)
        self.BN = nn.BatchNorm2d(filter_in, affine=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        out = self.lrule(self.BN(self.conv(input)))
        return out


class spatial_denoising(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(spatial_denoising, self).__init__()
        self.seqconv1=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling1=down_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv2=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling2=down_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv3=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling3=down_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv4=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.up_sampling1=up_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv5=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling2=up_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv6=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling3=up_sample_spatial(filter_in=64,filter_out=64,groups=1)

        self.seqconv7=Seq_conv(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.seqconv8=Seq_conv_tail(filter_in=64,filter_out=64, groups=groups)
 
        self.pred = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(3,3), stride=1, padding=(1,1), bias=False)

     def forward(self, input, shape):
        
        N,C,H,W=shape

        conc1 = self.seqconv1(input)

        out = self.down_sampling1(conc1, [N,C,H,W])

        conc2 = self.seqconv2(out)

        out = self.down_sampling2(conc2, [N,C,H//2,W//2])

        conc3 = self.seqconv3(out)

        out = self.down_sampling3(conc3, [N,C,H//4,W//4])

        out = self.seqconv4(out)

        out = self.up_sampling1(out, [N,C,H//8,W//8])

        out = torch.cat((out,conc3),1)

        out = self.seqconv5(out)

        out = self.up_sampling2(out, [N,C,H//4,W//4])

        out = torch.cat((out,conc2),1)

        out = self.seqconv6(out)

        out = self.up_sampling3(out, [N,C,H//2,W//2])

        out = torch.cat((out,conc1),1)

        out = self.seqconv7(out)

        out = self.seqconv8(out)

        out = self.pred(out)

        return out


class down_sample(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(down_sample, self).__init__()
        self.conv = nn.Conv3d(filter_in*4, filter_out , (1,1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,L,H,W=shape
        out = input.permute(0,2,1,3,4).reshape(N*L,-1,H,W)
        out = PixelUnshuffle.pixel_unshuffle(out,2)
        out = out.reshape(N,L,-1,H//2,W//2).permute(0,2,1,3,4)
        out = self.lrule(self.conv(out))

        return out


class up_sample(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(up_sample, self).__init__()
        self.conv = nn.Conv3d(filter_in, filter_out*4 , (1,1,1), 1, 0, groups=groups, bias=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input, shape):
        N,C,L,H,W=shape
        out = self.lrule(self.conv(input))
        out = out.permute(0,2,1,3,4).reshape(N*L,-1,H,W)
        out = F.pixel_shuffle(out,2)
        out = out.reshape(N,L,-1,H*2,W*2).permute(0,2,1,3,4)
        return out




class Seq_conv_ST(nn.Module):
     def __init__(self, filter_in=64,filter_out=64, concat_filter=64, groups=1):
        super(Seq_conv_ST, self).__init__()
        self.conv1 = nn.Conv3d(concat_filter, filter_out , (1,1,1), 1, 0, groups=groups, bias=True)
        self.conv2 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv2_T = nn.Conv3d(filter_in, filter_out , (3,1,1), 1, (1,0,0), groups=groups, bias=False)
        self.conv3 = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv3_T = nn.Conv3d(filter_in, filter_out , (3,3,3), 1, (1,1,1), groups=groups, bias=False)

        self.BN2 = nn.BatchNorm3d(filter_in, affine=True)
        self.BN3 = nn.BatchNorm3d(filter_in, affine=True)

        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        residual=self.conv1(input)
        out=self.lrule(self.conv2(residual))
        out=self.lrule(self.BN2(self.conv2_T(out)))
        out=self.lrule(self.conv3(out))
        out=self.lrule(self.BN3(self.conv3_T(out)))
        return out+residual


class Seq_conv_tail_ST(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(Seq_conv_tail_ST, self).__init__()
        self.conv = nn.Conv3d(filter_in, filter_out , (1,3,3), 1, (0,1,1), groups=groups, bias=False)
        self.conv_T = nn.Conv3d(filter_in, filter_out , (3,3,3), 1, (1,1,1), groups=groups, bias=False)
        self.BN = nn.BatchNorm3d(filter_in, affine=True)
        self.lrule = nn.LeakyReLU(negative_slope=0.1, inplace=True)

     def forward(self, input):
        out = self.lrule(self.conv(input))
        out = self.lrule(self.BN(self.conv_T(out)))
        return out


class spatiotemporal_denoising(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,groups=1):
        super(spatiotemporal_denoising, self).__init__()

        self.seqconv1=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling1=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv2=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling2=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv3=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.down_sampling3=down_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv4=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in, groups=groups)

        self.up_sampling1=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv5=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling2=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv6=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.up_sampling3=up_sample(filter_in=64,filter_out=64,groups=1)

        self.seqconv7=Seq_conv_ST(filter_in=64,filter_out=64, concat_filter=filter_in*2, groups=groups)

        self.seqconv8=Seq_conv_tail_ST(filter_in=64,filter_out=64, groups=groups)

        self.max_pooling1=nn.MaxPool3d((3,1,1),stride=(3,1,1),padding=(1,0,0))

        self.pred = pred(filter_in=64,filter_out=64,inchannel=3,groups=1)

     def forward(self, input, shape):
        
        N,C,L,H,W=shape

        conc1 = self.seqconv1(input)

        out = self.down_sampling1(conc1, [N,C,L,H,W])

        conc2 = self.seqconv2(out)

        out = self.down_sampling2(conc2, [N,C,L,H//2,W//2])

        conc3 = self.seqconv3(out)

        out = self.down_sampling3(conc3, [N,C,L,H//4,W//4])

        out = self.seqconv4(out)

        out = self.up_sampling1(out, [N,C,L,H//8,W//8])

        out = torch.cat((out,conc3),1)

        out = self.seqconv5(out)

        out = self.up_sampling2(out, [N,C,L,H//4,W//4])

        out = torch.cat((out,conc2),1)

        out = self.seqconv6(out)

        out = self.up_sampling3(out, [N,C,L,H//2,W//2])

        out = torch.cat((out,conc1),1)

        out = self.seqconv7(out)

        out = self.seqconv8(out)

        out = self.pred(out)

        return out



class pred(nn.Module):
     def __init__(self, filter_in=64,filter_out=64,inchannel=3,groups=1):
        super(pred, self).__init__()

        self.conv1 = nn.Conv3d(filter_out, filter_out , (3,3,3), (2,1,1), (0,1,1), groups=groups, bias=False)
        self.lrule1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.BN1 = nn.BatchNorm3d(filter_out, affine=True)

        self.Seq_conv1 = Seq_conv_ST(filter_in=filter_in,filter_out=filter_out, concat_filter=filter_in, groups=groups)

        self.conv2 = nn.Conv3d(filter_out, filter_out , (2,3,3), (2,1,1), (0,1,1), groups=groups, bias=False)

        self.lrule2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        self.BN2 = nn.BatchNorm3d(filter_out, affine=True)

        self.Seq_conv2 = Seq_conv_ST(filter_in=filter_out,filter_out=filter_out, concat_filter=filter_out, groups=groups)

        self.conv3 = nn.Conv3d(filter_out, 3 , (1,3,3), 1, (0,1,1), groups=groups, bias=False)

     def forward(self, input):
        out = self.lrule1(self.BN1(self.conv1(input)))  
        out = self.Seq_conv1(out)
        out = self.lrule2(self.BN2(self.conv2(out)))  
        out = self.Seq_conv2(out)
        out=  self.conv3(out) 

        return out



