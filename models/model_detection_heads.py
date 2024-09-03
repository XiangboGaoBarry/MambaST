import torch
import torch.nn as nn

class Detect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export

    def __init__(self, nc=80, anchors=(), num_frames=3, ch=()):  # detection layer
        super(Detect, self).__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.num_frames = num_frames

    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i])  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()




class LastFrameDetect(Detect):

    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            if len(x[i].shape) == 4:
                bs_f, c, h, w = x[i].shape
                x[i] = x[i].reshape(-1, self.num_frames, c, h, w).transpose(1, 2)
            x[i] = self.m[i](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            # x[i] = self.m[i](x[i].mean(dim=2))  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)
    
    
    
    



class MidFrameDetect(Detect):
    # TODO: Node that this detector meant to work for 3 frames only. 
    # May support more frames if necessary

    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            x[i] = self.m[i](x[i].permute(2,0,1,3,4)[1])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            # x[i] = self.m[i](x[i].mean(dim=2))  # conv
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                
                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

class FullFramesDetect(Detect):
    
    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        z = []  # inference output
        self.training |= self.export
        # x[i] = 
        for i in range(self.nl):
            x[i] = x[i].permute(0,2,1,3,4)
            # flatten the frame and batch dimensions (bs, nf, c, h, w) -> (bs*nf, c, h, w)
            bs, nf, c, h, w = x[i].shape
            x[i] = x[i].reshape(bs * nf, c, h, w)
            x[i] = self.m[i](x[i])
            _, _, ny, nx = x[i].shape  
            # x(bs*nf,255,20,20) to x(bs*nf,3,20,20,85)
            x[i] = x[i].view(bs * nf, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh
                z.append(y.view(bs, nf, -1, self.no))

            x[i] = x[i].view(bs, nf, self.na, ny, nx, self.no)

        return x if self.training else (torch.cat(z, 2), x, feature)
    
class ThermalRgbDetect(nn.Module):
    stride = None  # strides computed during build
    export = False  # onnx export
    def __init__(self, nc=80, anchors=(), ch=()):  # detection layer
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)*2 # number of detection layers is doubled because thermal and RGB
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.zeros(1)] * self.nl  # init grid
        a = torch.tensor(anchors).float().view(self.nl//2, -1, 2)
        self.register_buffer('anchors', a)  # shape(nl,na,2)
        self.register_buffer('anchor_grid', a.clone().view(self.nl//2, 1, -1, 1, 1, 2))  # shape(nl,1,na,1,1,2)
        self.conv_modules_init(ch)
        self.fuseconv_init(ch)

    def conv_modules_init(self, ch):
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for RGB
        self.m_ir = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv for Thermal

    def fuseconv_init(self, ch):
        self.conv = nn.ModuleList(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0) for dim in ch)
        self.conv_ir = nn.ModuleList(nn.Conv3d(in_channels=dim*2, out_channels=dim ,kernel_size=1, stride=1, padding=0) for dim in ch)
    
    def fuseconv(self, x):
        fused_bb_rgb_thermal = x[:3]
        fused_in_head =  x[3:]
        out = []
        out_ir = []
        for i, xx in enumerate(zip(fused_bb_rgb_thermal)):
            out.append(self.conv[i](torch.cat((xx[0][0], fused_in_head[i]),dim=1))) # RGB
            out_ir.append(self.conv_ir[i](torch.cat((xx[0][1], fused_in_head[i]),dim=1))) # Thermal
        out.extend(out_ir)
        return out
    
    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        x = self.fuseconv(x)
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            j = i % 3 # first 3 are RGB, last 3 are thermal
            if i < 3:
                x[i] = self.m[j](x[i])  # conv
            else:
                x[i] = self.m_ir[j](x[i])  # conv

            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

class LastFrameThermalRgbDetect(ThermalRgbDetect):
    def forward(self, x):
        feature = x.copy()
        # x = x.copy()  # for profiling
        x = self.fuseconv(x)
        z = []  # inference output
        self.training |= self.export
        for i in range(self.nl):
            j = i % 3 # first 3 are RGB, last 3 are thermal
            if i < 3:
                x[i] = self.m[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
            else:
                x[i] = self.m_ir[j](x[i].permute(2,0,1,3,4)[0])  # conv, (using [0] because in dataloader getitem used '::-1' and then reshaped)
                
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if not self.training:  # inference
                if self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i] = self._make_grid(nx, ny).to(x[i].device )

                y = x[i].sigmoid()
                y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i]) * self.stride[j]  # xy
                y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[j]  # wh
                z.append(y.view(bs, -1, self.no))

        return x if self.training else (torch.cat(z, 1), x, feature)
