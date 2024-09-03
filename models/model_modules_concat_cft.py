# import torch
# import torch.nn as nn
# from .concat_cft_conv import Conv

# class Focus(nn.Module):
#     # Focus wh information into c-space
#     def __init__(self, c1, c2, nf, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, (nf) Number of Local + Global Frames in Batch, kernel, stride, padding, groups
#         super(Focus, self).__init__()
#         # print("c1 * 4, c2, k", c1 * 4, c2, k)
#         self.conv = Conv(c1 * 4 * nf, c2, k, s, p, g, act)
#         # self.contract = Contract(gain=2)

#     def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
#         # print("Focus inputs shape", x.shape)
#         # print()
#         return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


# class Bottleneck(nn.Module):
#     # Standard bottleneck
#     def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
#         super(Bottleneck, self).__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_, c2, 3, 1, g=g)
#         self.add = shortcut and c1 == c2

#     def forward(self, x):
#         return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

# class Add2(nn.Module):
#     #  x + transformer[0] or x + transformer[1]
#     def __init__(self, c1, index):
#         super().__init__()
#         self.index = index

#     def forward(self, x):
#         if self.index == 0:
#             return torch.add(x[0], x[1][0])
#         elif self.index == 1:
#             return torch.add(x[0], x[1][1])
#         # return torch.add(x[0], x[1])

# class SPP(nn.Module):
#     # Spatial pyramid pooling layer used in YOLOv3-SPP
#     def __init__(self, c1, c2, k=(5, 9, 13)):
#         super(SPP, self).__init__()
#         c_ = c1 // 2  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
#         self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

#     def forward(self, x):
#         x = self.cv1(x)
#         return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))



import torch
import torch.nn as nn
from .concat_cft_conv import Conv

class Upsample(nn.Module):
    def __init__(self, size, scale_factor, mode):
        super().__init__()
        self.upsample = nn.Upsample(size, scale_factor, mode)
        self.sf = scale_factor
    def forward(self, x):
        b,c, num_frames, w, h = x.shape
        x = x.reshape(b,-1,w,h)
        return self.upsample(x).reshape(b,c,num_frames, w*self.sf, h*self.sf)

class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, nf, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, (nf) Number of Local + Global Frames in Batch, kernel, stride, padding, groups
        super(Focus, self).__init__()
        # print("c1 * 4, c2, k", c1 * 4, c2, k)
        self.conv = Conv(c1 * 4 * nf, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # print("Focus inputs shape", x.shape)
        # print()
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super(Bottleneck, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

class Add2(nn.Module):
    #  x + transformer[0] or x + transformer[1]
    def __init__(self, c1, index):
        super().__init__()
        self.index = index

    def forward(self, x):
        if self.index == 0:
            return torch.add(x[0], x[1][0])
        elif self.index == 1:
            return torch.add(x[0], x[1][1])
        # return torch.add(x[0], x[1])

class SPP(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super(SPP, self).__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))

