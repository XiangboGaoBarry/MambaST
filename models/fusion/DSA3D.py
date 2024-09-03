import torch.nn as nn
import torch.nn.functional as F
import torch
from models.fusion.deformable_attention import deformable_attn_pytorch_3d, \
    LearnedPositionalEncoding, constant_init, xavier_init
import math
import warnings
from utils.torch_utils import time_synchronized

class DeformableSpatialAttentionLayer3D(nn.Module):
    def __init__(self, 
                 embed_dims,
                 num_heads=18,
                 num_points=12,
                 num_frames=3,
                 dropout=0.1):
        super(DeformableSpatialAttentionLayer3D, self).__init__()
        if embed_dims % num_heads != 0:
            warnings.warn(
                "The number of heads in MultiScaleDeformAttention "
                'is not a multiple of 18, which may be suboptimal.')
            self.value_dims = embed_dims // num_heads * num_heads
        else:
            self.value_dims = embed_dims
        
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_frames = num_frames
        self.dropout = nn.Dropout(dropout)
        self.sampling_offsets = nn.Linear(self.embed_dims, num_heads * num_points * 3)
        self.attention_weights = nn.Linear(self.embed_dims, num_heads * num_points)
        self.value_proj = nn.Linear(self.embed_dims, self.value_dims)
        self.output_proj = nn.Linear(self.value_dims, self.embed_dims)
        self.init_weights()
    
    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        theta = torch.arange(8) * (2.0 * math.pi / 8)

        yaw_init = torch.stack([theta.cos(), theta.sin(), torch.zeros(8)], -1)
        roll_init = torch.stack([torch.zeros(8), theta.cos(), theta.sin()], -1)
        pitch_init = torch.stack([theta.cos(), torch.zeros(8), theta.sin()], -1)    
        grid_init = torch.cat([yaw_init, pitch_init[[1,2,3,5,6,7]], roll_init[[1,3,5,7]]], 0)
        
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, 3).repeat(1, 1, self.num_points, 1)

        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        # TODO: Remove the hard coded half precision
        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        
    def forward(self, 
                query,
                key=None,
                value=None,
                query_pos=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None,):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                (bs, num_query, embed_dims).
            value (Tensor): The value tensor with shape
                (bs, num_query, embed_dims).
            spatial_shapes (tuple): Spatial shape of features (h, w).

        Returns:
             Tensor: forwarded results with shape [bs, num_query, embed_dims].
        """
        
        bs, num_query, _ = query.shape
        h, w = spatial_shapes
        
        
        if query_pos is not None:
            query = query + query_pos
        value = self.value_proj(value)
        # if key_padding_mask is not None:
        #     value = value.masked_fill(key_padding_mask[..., None], 0.0)
        D = self.num_frames * 2 - 1
        value = value.reshape(bs, num_query * D, self.num_heads, self.value_dims//self.num_heads) # bs, num_query, num_heads, embed_dims//num_heads
        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_points, 3) # bs, num_query, num_heads, num_points, 2
        attention_weights = self.attention_weights(query).view(bs, num_query, self.num_heads, self.num_points) # bs, num_query, num_heads, num_points
        attention_weights = attention_weights.softmax(-1).to(dtype) # TODO: attention_weights.softmax(-1) changed attention_weights from half to float
        
        reference_points = self.get_reference_points(h, w, bs=bs, device=device, dtype=dtype) # bs, num_query, 3
        offset_normalizer = torch.Tensor([w, h, self.num_frames]).to(device).to(dtype)
        sampling_locations = reference_points[:, :, None, None, :] \
            + sampling_offsets / offset_normalizer
            
        # sampling_offsets
        
        output = self.output_proj(deformable_attn_pytorch_3d(value, (h, w, D), sampling_locations, attention_weights))
        
        return self.dropout(output)
        
    
    def get_reference_points(self, H, W, bs=1, device='cuda', dtype=torch.half):
        ref_y, ref_x = torch.meshgrid(
                torch.linspace(
                    0.5, H - 0.5, H, dtype=dtype, device=device),
                torch.linspace(
                    0.5, W - 0.5, W, dtype=dtype, device=device),
            )
        ref_y = ref_y.reshape(-1)[None] / H
        ref_x = ref_x.reshape(-1)[None] / W
        ref_3d = torch.stack((ref_x, ref_y, torch.zeros(H*W)[None].to(device)), -1)
        ref_3d = ref_3d.repeat(bs, 1, 1)
        return ref_3d

class DeformableSpatialAttentionModule3D(nn.Module):
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=18,
                 num_points=12,
                 num_frames=3,
                 dropout=0.1):
        super(DeformableSpatialAttentionModule3D, self).__init__()
        self.embed_dims = embed_dims
        self.num_frames = num_frames
        self.positional_encoding = LearnedPositionalEncoding(embed_dims//2, H, W)
        self.attention_layers = nn.ModuleList()
        for _ in range(n_layers):
            self.attention_layers.append(DeformableSpatialAttentionLayer3D(embed_dims, num_heads, num_points, num_frames, dropout))
    
    def forward(self,
                layer,
                query,
                key=None,
                value=None,
                device='cuda',
                dtype=torch.half,
                spatial_shapes=None):
        # bs, embed_dims, num_frames, h, w
        bs = query.shape[0]
        h, w = spatial_shapes
        pos_mask = torch.zeros((bs, h, w), device=device).to(dtype)
        query_pos = self.positional_encoding(pos_mask).to(dtype).flatten(2).transpose(1,2) # bs, num_query, embed_dims=pos_dim*2
        value = torch.cat([query, value[:, :, :-1].flip(dims=[2])], dim=2).reshape(bs, self.embed_dims, -1).transpose(1, 2) # bs, num_query, embed_dims
        
        last_query = query[:, :, -1].reshape(bs, self.embed_dims, -1).transpose(1, 2) # bs, num_query, embed_dims
        out = self.attention_layers[layer](query=last_query,
                                            key=last_query,
                                            value=value,
                                            query_pos=query_pos,
                                            device=device,
                                            dtype=dtype,
                                            spatial_shapes=spatial_shapes)
        query = torch.cat([query[:, :, :-1], query[:, :, -1:] + out.transpose(1, 2).reshape(bs, self.embed_dims, 1, h, w)], dim=2)
        return query
    
    
    
class DeformableSpatialAttention3D(nn.Module):
    
    def __init__(self, 
                 embed_dims,
                 H,
                 W,
                 n_layers=8,
                 num_heads=18,
                 num_points=12,
                 num_frames=3,
                 dropout=0.1):
        super(DeformableSpatialAttention3D, self).__init__()
        self.n_layers = n_layers
        self.embed_dims = embed_dims
        self.H = H
        self.W = W
        self.n_layers = n_layers
        self.num_heads = num_heads
        self.num_points = num_points
        self.dropout = dropout
        self.num_frames = num_frames
        self.initModule()
        
    def initModule(self):
        self.rgb_attention = DeformableSpatialAttentionModule3D(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.num_frames, self.dropout)
        self.ir_attention = DeformableSpatialAttentionModule3D(self.embed_dims, self.H, self.W, self.n_layers, self.num_heads, self.num_points, self.num_frames, self.dropout)
        
    def forward(self, x):
        '''
        Args:
            x (tuple)
        return:
        '''
        import time
        starttime = time_synchronized()
        rgb_fea = x[0]
        ir_fea = x[1]
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        bs, embed_dims, num_frames, h, w = rgb_fea.shape
        # rgb_fea = rgb_fea.reshape(bs, self.embed_dims, self.num_frames * self.H * self.W).transpose(1, 2)  # bs, nf*h*w, embed_dims
        # ir_fea = ir_fea.reshape(bs, self.embed_dims, self.num_frames * self.H * self.W).transpose(1, 2)  # bs, nf*h*w, embed_dims
        
        for layer in range(self.n_layers):
            # try:
            rgb_fea_out = self.rgb_attention(layer=layer,
                                            query=rgb_fea, 
                                            key=None, 
                                            value=ir_fea, 
                                            device=rgb_fea.device, 
                                            dtype=rgb_fea.dtype,
                                            spatial_shapes=(h, w))
                
                
            # except Exception as e:
            #     print('rgb_fea')
            #     import traceback; traceback.print_exc()
            #     import pdb; pdb.set_trace()
                
            # try:
            ir_fea_out = self.ir_attention(layer=layer,
                                            query=ir_fea, 
                                            key=None, 
                                            value=rgb_fea, 
                                            device=ir_fea.device,
                                            dtype=ir_fea.dtype,
                                            spatial_shapes=(h, w))
            
            # except Exception as e:
            #     print('ir_fea')
            #     import traceback; traceback.print_exc()
            #     import pdb; pdb.set_trace()

            
            rgb_fea = rgb_fea_out
            ir_fea = ir_fea_out
        
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        # rgb_fea = rgb_fea.transpose(1,2).reshape(bs, self.embed_dims, self.num_frames, self.H, self.W)
        # ir_fea = ir_fea.transpose(1,2).reshape(bs, self.embed_dims, self.num_frames, self.H, self.W)
        print("Time: ", time_synchronized()-starttime)
        return rgb_fea, ir_fea
        