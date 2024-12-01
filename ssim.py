import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class L1_DSSIM_Loss(Module):
    def __init__(self, alpha=0.5):
        super(L1_DSSIM_Loss, self).__init__()
        self.alpha = alpha 
     
    def forward(self, output, target):
        ssim_module = SSIM(data_range=255, size_average=True, channel=3) # channel=1 for grayscale images
        l1_loss = F.l1_loss(output, target)
        dssim_loss = 1 - ssim_module(output, target)
        return self.alpha * l1_loss + (1 - self.alpha) * dssim_loss