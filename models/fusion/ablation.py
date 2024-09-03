import torch.nn as nn
import torch.nn.functional as F
import torch

class FeatAddTemporal(nn.Module):
    """  FeatConcat """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, c, num_frames, h, w = rgb_fea.shape
        feat = rgb_fea.sum(dim=2) + ir_fea.sum(dim=2)
        rgb_fea = feat.repeat(1, 1, num_frames, 1, 1)
        ir_fea = feat.repeat(1, 1, num_frames, 1, 1)
        return rgb_fea, ir_fea
    

class FeatAdd(nn.Module):
    """  FeatConcat """

    def __init__(self, nf):
        super().__init__()
        self.nf = nf

    def forward(self, x):
        """
        Args:
            x (tuple?)

        """
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, H, W)
        ir_fea = x[1]   # ir_fea (tensor): dim:(B, C, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs_num_frames, c, h, w = rgb_fea.shape
        # feat = rgb_fea.sum(dim=2) + ir_fea.sum(dim=2)
        
        rgb_fea = rgb_fea.reshape(bs_num_frames//self.nf, self.nf, c, h, w).sum(dim=1).repeat(self.nf, 1, 1, 1)
        ir_fea = ir_fea.reshape(bs_num_frames//self.nf, self.nf, c, h, w).sum(dim=1).repeat(self.nf, 1, 1, 1)
        return rgb_fea, ir_fea    
    

class FeatIdentity(nn.Module):
    def __init__(self):
        super(FeatIdentity, self).__init__()

    def forward(self, x):
        return x
    
    