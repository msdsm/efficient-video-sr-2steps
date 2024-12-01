import torch
import torch.nn as nn
from unet_modules import ConvPair, UpSamplingBlock, DownSamplingBlock, OutConvBlock

class Unet(nn.Module):
    def __init__(self, bilinear=False):
        super(Unet, self).__init__()
        self.convp = ConvPair(in_channels=3, out_channels=64)
        self.down1 = DownSamplingBlock(64, 128)
        self.down2 = DownSamplingBlock(128, 256)
        self.down3 = DownSamplingBlock(256, 512)
        self.down4 = DownSamplingBlock(512, 1024)
        self.down5 = DownSamplingBlock(1024, 1024)

        self.up1 = UpSamplingBlock(1024, 512)
        self.up2 = UpSamplingBlock(512, 256)
        self.up3 = UpSamplingBlock(256, 128)
        self.up4 = UpSamplingBlock(128, 64)
        self.out = OutConvBlock(64, 3)

        self.encoder_l = UnetEncoder()
        self.encoder_r = UnetEncoder()

    def forward(self, x, x_l, x_r):
        x1 = self.convp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x5 = self.down5(x5)
        
        img_feature_l = self.encoder_l(x_l)
        img_feature_r = self.encoder_r(x_r)
        x5 = x5+img_feature_l+img_feature_r

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.out(x)

        # return x, x5, img_feature_l, img_feature_r
        return x


class UnetEncoder(nn.Module):
    def __init__(self, bilinear=False):
        super(UnetEncoder, self).__init__()
        self.convp = ConvPair(in_channels=3, out_channels=64)
        self.down1 = DownSamplingBlock(64, 128)
        self.down2 = DownSamplingBlock(128, 256)
        self.down3 = DownSamplingBlock(256, 512)
        self.down4 = DownSamplingBlock(512, 1024)

    def forward(self, x):
        x1 = self.convp(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return x5