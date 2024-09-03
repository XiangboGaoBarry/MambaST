import torch
import torch.nn as nn
import torch.nn.functional as F

class LearnedPositionalEncoding(nn.Module):
    """Position embedding with learnable embedding weights.

    Args:
        num_feats (int): The feature dimension for each position
            along x-axis or y-axis. The final returned dimension for
            each position is 2 times of this value.
        row_num_embed (int, optional): The dictionary size of row embeddings.
            Default 50.
        col_num_embed (int, optional): The dictionary size of col embeddings.
            Default 50.
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 num_feats,
                 row_num_embed=50,
                 col_num_embed=50):
        super(LearnedPositionalEncoding, self).__init__()
        self.row_embed = nn.Embedding(row_num_embed, num_feats)
        self.col_embed = nn.Embedding(col_num_embed, num_feats)
        self.num_feats = num_feats
        self.row_num_embed = row_num_embed
        self.col_num_embed = col_num_embed

    def forward(self, mask):
        """Forward function for `LearnedPositionalEncoding`.

        Args:
            mask (Tensor): ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image. Shape [bs, h, w].

        Returns:
            pos (Tensor): Returned position embedding with shape
                [bs, num_feats*2, h, w].
        """
        h, w = mask.shape[-2:]
        x = torch.arange(w, device=mask.device)
        y = torch.arange(h, device=mask.device)
        x_embed = self.col_embed(x)
        y_embed = self.row_embed(y)
        pos = torch.cat(
            (x_embed.unsqueeze(0).repeat(h, 1, 1), 
             y_embed.unsqueeze(1).repeat(1, w, 1)),
            dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(mask.shape[0], 1, 1, 1)
        return pos


    def __repr__(self):
        """str: a string that describes the module"""
        repr_str = self.__class__.__name__
        repr_str += f'(num_feats={self.num_feats}, '
        repr_str += f'row_num_embed={self.row_num_embed}, '
        repr_str += f'col_num_embed={self.col_num_embed})'
        return repr_str
    
    

def deformable_attn_pytorch(
        value: torch.Tensor, value_spatial_shapes: torch.Tensor,
        sampling_locations: torch.Tensor,
        attention_weights: torch.Tensor) -> torch.Tensor:
    """non-cuda version of deformable attention.

    Args:
        value (torch.Tensor): The value has shape
            (bs, num_keys, num_heads, embed_dims//num_heads)
        value_spatial_shapes (tuple[int]): The spatial shape of value (H, W)
        sampling_locations (torch.Tensor): The location of sampling points,
            has shape (bs ,num_queries, num_heads, num_points, 2),
            the last dimension 2 represent (x, y).
        attention_weights (torch.Tensor): The weight of sampling points used
            when calculate the attention, has shape
            (bs ,num_queries, num_heads, num_points),

    Returns:
        torch.Tensor: has shape (bs, num_queries, embed_dims)
    """
    
    dtype = value.dtype
    bs, _, num_heads, embed_dims = value.shape
    _, num_queries, num_heads, num_points, _ =\
        sampling_locations.shape
    sampling_grids = 2 * sampling_locations - 1
    H_, W_ = value_spatial_shapes
    # bs, H_*W_, num_heads, embed_dims ->
    # bs, H_*W_, num_heads*embed_dims ->
    # bs, num_heads*embed_dims, H_*W_ ->
    # bs*num_heads, embed_dims, H_, W_
    value = value.flatten(2).transpose(1, 2).reshape(
        bs * num_heads, embed_dims, H_, W_)
    # bs, num_queries, num_heads, num_points, 2 ->
    # bs, num_heads, num_queries, num_points, 2 ->
    # bs*num_heads, num_queries, num_points, 2
    sampling_grids = sampling_grids.transpose(1, 2).flatten(0, 1)
    # bs*num_heads, embed_dims, num_queries, num_points
    sampling_values = F.grid_sample(
        value,
        sampling_grids,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False)

    # (bs, num_queries, num_heads, num_points) ->
    # (bs, num_heads, num_queries, num_points) ->
    # (bs, num_heads, 1, num_queries, num_points)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        bs * num_heads, 1, num_queries, num_points)
    output = (sampling_values * attention_weights).sum(-1).to(dtype).view(bs, num_heads * embed_dims,
                                              num_queries)
    return output.transpose(1, 2).contiguous()

def constant_init(module: nn.Module, val: float, bias: float = 0) -> None:
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)
        
def xavier_init(module: nn.Module,
                gain: float = 1,
                bias: float = 0,
                distribution: str = 'normal') -> None:
    assert distribution in ['uniform', 'normal']
    if hasattr(module, 'weight') and module.weight is not None:
        if distribution == 'uniform':
            nn.init.xavier_uniform_(module.weight, gain=gain)
        else:
            nn.init.xavier_normal_(module.weight, gain=gain)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)