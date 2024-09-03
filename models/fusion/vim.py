# from models.fusion.Mamba import Mamba
from models.fusion.functools import partial
from collections import namedtuple

import math
import random
import torch
import torch.nn as nn
from functools import partial
from torch import Tensor
from typing import Optional
from timm.models.layers import trunc_normal_, lecun_normal_
from timm.models.layers import DropPath, to_2tuple
from timm.models.vision_transformer import _load_weights
from utils.torch_utils import time_synchronized
from mamba_ssm.modules.mamba_simple import Mamba

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class PatchEmbed(nn.Module):
    """2D Image to Patch Embedding"""

    def __init__(self, img_size, patch_size=(2, 2), stride=(2, 2), in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (
            (self.img_size[0] - patch_size[0]) // stride[0] + 1,
            (self.img_size[1] - patch_size[1]) // stride[1] + 1,
        )
        self.num_patches = self.grid_size[0] * self.grid_size[1]

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        # self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        # self.norm = norm_layer
        self.norm = nn.LayerNorm((embed_dim, self.grid_size[0], self.grid_size[1]), eps=1e-6)

    def forward(self, x):
        B, C, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        x = self.norm(x)
        return x


class Block(nn.Module):
    def __init__(
        self,
        dim,
        mixer_cls,
        norm_cls=nn.LayerNorm,
        fused_add_norm=False,
        residual_in_fp32=False,
        drop_path=0.0,
    ):
        """
        Simple block wrapping a mixer class with LayerNorm/RMSNorm and residual connection"

        This Block has a slightly different structure compared to a regular
        prenorm Transformer block.
        The standard block is: LN -> MHA/MLP -> Add.
        [Ref: https://arxiv.org/abs/2002.04745]
        Here we have: Add -> LN -> Mixer, returning both
        the hidden_states (output of the mixer) and the residual.
        This is purely for performance reasons, as we can fuse add and LayerNorm.
        The residual needs to be provided (except for the very first block).
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, "RMSNorm import fails"
            assert isinstance(
                self.norm, (nn.LayerNorm, RMSNorm)
            ), "Only LayerNorm and RMSNorm are supported for fused_add_norm"

    def forward(self, hidden_states: Tensor, residual: Optional[Tensor] = None, inference_params=None, prev_state=None):
        r"""Pass the input through the encoder layer.

        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: hidden_states = Mixer(LN(residual))
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        y, hidden_states = self.mixer(hidden_states, inference_params=inference_params, prev_state=prev_state)
        return y, hidden_states, residual

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)


def create_block(
    d_model,
    ssm_cfg=None,
    norm_epsilon=1e-5,
    drop_path=0.0,
    rms_norm=False,
    residual_in_fp32=False,
    fused_add_norm=False,
    layer_idx=None,
    device=None,
    dtype=None,
    if_bimamba=False,
    bimamba_type="none",
    if_devide_out=False,
    init_layer_scale=None,
):
    if if_bimamba:
        bimamba_type = "v1"
    if ssm_cfg is None:
        ssm_cfg = {}
    factory_kwargs = {"device": device, "dtype": dtype}
    mixer_cls = partial(
        Mamba,
        layer_idx=layer_idx,
        bimamba_type=bimamba_type,
        if_devide_out=if_devide_out,
        init_layer_scale=init_layer_scale,
        use_fast_path=False,
        **ssm_cfg,
        **factory_kwargs,
    )
    norm_cls = partial(nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_epsilon, **factory_kwargs)
    block = Block(
        d_model,
        mixer_cls,
        norm_cls=norm_cls,
        drop_path=drop_path,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454
def _init_weights(
    module,
    n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            if not getattr(module.bias, "_no_reinit", False):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        for name, p in module.named_parameters():
            if name in ["out_proj.weight", "fc2.weight"]:
                # Special Scaled Initialization --> There are 2 Layer Norms per Transformer Block
                # Following Pytorch init, except scale by 1/sqrt(2 * n_layer)
                # We need to reinit p since this code could be called multiple times
                # Having just p *= scale would repeatedly scale it down
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def segm_init_weights(m):
    if isinstance(m, nn.Linear):
        trunc_normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


class VIMModule(nn.Module):

    def __init__(
        self,
        embed_dims,
        n_layers,
        inter_dpr,
        ssm_cfg=None,
        norm_epsilon=1e-5,
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        if_bimamba=False,
        bimamba_type="none",
        if_devide_out=False,
        init_layer_scale=None,
        initializer_cfg=None,
        device=None,
        dtype=None,
        factory_kwargs={},
    ):

        super(VIMModule, self).__init__()

        self.layers = nn.ModuleList(
            [
                create_block(
                    embed_dims,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    if_bimamba=if_bimamba,
                    bimamba_type=bimamba_type,
                    drop_path=inter_dpr[i],
                    if_devide_out=if_devide_out,
                    init_layer_scale=init_layer_scale,
                    **factory_kwargs,
                )
                for i in range(n_layers)
            ]
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layers,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token", "dist_token", "cls_token_head", "cls_token_tail"}

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=""):
        _load_weights(self, checkpoint_path, prefix)

    def forward(self, hidden_states, residual, layer, prev_state=None):
        y, hidden_states, residual = self.layers[layer](hidden_states, residual, prev_state=prev_state)
        return y, hidden_states, residual


class VIMAttention(nn.Module):
    """Deformable Attention Module."""

    def __init__(
        self,
        embed_dims,
        H,
        W,
        n_layers=8,
        patch_size=10,
        stride=10,
        num_frames=3,
        fused_add_norm=True,
        channels=3,
        norm_epsilon=1e-5,
        #  dropout=0.1,
        drop_path_rate=0.1,
        drop_rate=0.1,
        device=None,
        dtype=None,
        if_abs_pos_embed=True,
        factory_kwargs={},
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super(VIMAttention, self).__init__()
        self.n_layers = n_layers
        self.embed_dims = embed_dims
        assert H == W, "H must be equal to W for current implementation"
        self.H = H
        self.W = W
        self.patch_size = patch_size
        self.stride = stride
        self.channels = channels
        # self.dropout = dropout
        self.num_frames = num_frames
        self.norm_epsilon = norm_epsilon
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        fused_add_norm = True
        self.if_abs_pos_embed = if_abs_pos_embed
        self.fused_add_norm = fused_add_norm

        self.num_tokens = 0

        self.initModule()

    def initModule(self):
        # self.patch_embed = PatchEmbed(
        #     img_size=self.H, patch_size=self.patch_size, stride=self.stride, in_chans=self.channels, embed_dim=self.embed_dims)
        # num_patches = self.patch_embed.num_patches
        num_patches = self.H * self.W * self.num_frames

        self.norm_f = nn.LayerNorm(self.embed_dims, eps=self.norm_epsilon, **self.factory_kwargs)
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.n_layers)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()
        self.attention = VIMModule(
            embed_dims=self.embed_dims, n_layers=self.n_layers, inter_dpr=inter_dpr, fused_add_norm=self.fused_add_norm
        )

        if self.if_abs_pos_embed:
            self.pos_embed = nn.Parameter(torch.zeros(1, num_patches * 2 + self.num_tokens, self.embed_dims))
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        if self.if_abs_pos_embed:
            trunc_normal_(self.pos_embed, std=0.02)

        # self.patch_embed.apply(segm_init_weights)

    def forward(self, x):
        # import time
        # starttime = time_synchronized()
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, num_frames, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, num_frames, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]
        # bs_org, c_org, num_frames, h, w = rgb_fea.shape
        # rgb_fea = rgb_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        # ir_fea = ir_fea.transpose(1,2).reshape(bs_org*num_frames, c_org, h,w)
        bs, embed_dims, num_frames, h, w = rgb_fea.shape

        # rgb_fea = rgb_fea.reshape(bs, embed_dims, h*w).transpose(1, 2)  # bs, h*w, embed_dims
        # ir_fea = ir_fea.reshape(bs, embed_dims, h*w).transpose(1, 2)  # bs, h*w, embed_dims

        rgb_fea = rgb_fea.reshape(bs, embed_dims, num_frames * h * w).transpose(1, 2)  # bs, h*w, embed_dims
        ir_fea = ir_fea.reshape(bs, embed_dims, num_frames * h * w).transpose(1, 2)  # bs, h*w, embed_dims

        x = torch.cat((rgb_fea, ir_fea), dim=1)
        B, M, _ = x.shape
        # x = self.patch_embed(x)

        if self.if_abs_pos_embed:
            x = x + self.pos_embed
            x = self.pos_drop(x)

        residual = None
        hidden_states = x
        for i in range(self.n_layers // 2):
            _, hidden_states_f, residual_f = self.attention(hidden_states, residual, layer=i * 2)
            _, hidden_states_b, residual_b = self.attention(
                hidden_states.flip([1]), None if residual == None else residual.flip([1]), layer=i * 2 + 1
            )
            hidden_states = hidden_states_f + hidden_states_b.flip([1])
            residual = residual_f + residual_b.flip([1])
        if residual is None:
            residual = hidden_states
        else:
            residual = residual + self.drop_path(hidden_states)
        hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))

        rgb_fea, ir_fea = hidden_states.split([num_frames * h * w, num_frames * h * w], dim=1)
        # bs, num_query, embed_dims -> bs_org, channels, num_frames, h,w
        rgb_fea = rgb_fea.transpose(1, 2).reshape(bs, embed_dims, num_frames, h, w)
        ir_fea = ir_fea.transpose(1, 2).reshape(bs, embed_dims, num_frames, h, w)
        # print("Time: ", time_synchronized()-starttime)
        return rgb_fea, ir_fea


class VIMAttentionV2(nn.Module):
    """Deformable Attention Module."""

    def __init__(
        self,
        embed_dims,
        H,
        W,
        n_layers=8,
        num_frames=3,
        mhh_patches=4,  # number multi-head hierarchical patching
        fused_add_norm=True,
        order_concat_flatten=True,
        dual_scan=True,
        norm_epsilon=1e-5,
        #  dropout=0.1,
        drop_path_rate=0.1,
        drop_rate=0.1,
        device=None,
        dtype=None,
        if_abs_pos_embed=True,
        factory_kwargs={},
    ):
        self.factory_kwargs = {"device": device, "dtype": dtype}
        super(VIMAttentionV2, self).__init__()
        self.n_layers = n_layers
        self.embed_dims = embed_dims
        self.dual_scan = dual_scan
        assert H == W, "H must be equal to W for current implementation"
        assert (
            mhh_patches > 0 and (mhh_patches & (mhh_patches - 1)) == 0
        ), "mhh_patches is recommanded to be power of 2, for better embedding dim split"
        assert embed_dims % mhh_patches == 0, "embed_dims must be divisible by mhh_patches"
        self.embed_dims_mhh = self.embed_dims // mhh_patches
        if self.dual_scan:
            assert self.embed_dims_mhh % 2 == 0, "embed_dims_mhh must be divisible by 2 for dual path"
            self.embed_dims_mhh = self.embed_dims_mhh // 2
        self.patch_factors_list = [2**i for i in range(mhh_patches)]
        self.H = H
        self.W = W
        self.order_concat_flatten = order_concat_flatten
        # self.dropout = dropout
        self.num_frames = num_frames
        self.norm_epsilon = norm_epsilon
        self.drop_path_rate = drop_path_rate
        self.drop_rate = drop_rate
        self.if_abs_pos_embed = if_abs_pos_embed
        self.fused_add_norm = fused_add_norm

        self.num_tokens = 0

        self.initModule()

    def initModule(self):
        # num_patches = self.H * self.W * self.num_frames

        self.norm_f = nn.LayerNorm(self.embed_dims_mhh, eps=self.norm_epsilon, **self.factory_kwargs)

        patch_embeding_list = [
            PatchEmbed(
                img_size=(self.H, self.W),
                patch_size=(s, s),
                stride=(s, s),
                in_chans=self.embed_dims,
                embed_dim=self.embed_dims_mhh,
            )
            for s in self.patch_factors_list
        ]
        num_patches_list = [p.num_patches for p in patch_embeding_list]

        self.patch_embeding_list = nn.ModuleList(patch_embeding_list)

        if self.dual_scan:
            patch_embeding_reversed_list = [
                PatchEmbed(
                    img_size=(self.H, self.W),
                    patch_size=(s, s),
                    stride=(s, s),
                    in_chans=self.embed_dims,
                    embed_dim=self.embed_dims_mhh,
                )
                for s in self.patch_factors_list
            ]
            self.patch_embeding_reversed_list = nn.ModuleList(patch_embeding_reversed_list)

        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, self.n_layers)]
        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(self.drop_path_rate) if self.drop_path_rate > 0.0 else nn.Identity()
        self.attention_list = [
            VIMModule(
                embed_dims=self.embed_dims_mhh,
                n_layers=self.n_layers,
                inter_dpr=inter_dpr,
                fused_add_norm=self.fused_add_norm,
            )
            for _ in range(len(num_patches_list))
        ]
        self.attention_list = nn.ModuleList(self.attention_list)

        if self.if_abs_pos_embed:
            self.pos_embed = [
                nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, self.embed_dims_mhh))
                for num_patches in num_patches_list
            ]
            self.pos_embed = nn.ParameterList(self.pos_embed)
            self.pos_drop = nn.Dropout(p=self.drop_rate)

        if self.if_abs_pos_embed:
            for pos_emb in self.pos_embed:
                trunc_normal_(pos_emb, std=0.02)

        self.up_sample_list = [
            nn.Upsample(scale_factor=2**i, mode="bilinear", align_corners=False) for i in range(len(num_patches_list))
        ]
        self.up_sample_list = nn.ModuleList(self.up_sample_list)

        self.modality_embed = nn.Parameter(torch.zeros(2, self.embed_dims_mhh))
        trunc_normal_(self.modality_embed, std=0.02)

        # self.patch_embed.apply(segm_init_weights)

    def perform_order_concat_flatten(self, x):
        indices = torch.arange(x.size(2))
        bs_num_frames, embed_dims_mhh, patch_h, patch_w = x.shape
        flip_mask = indices % 2 == 1
        flipped_rows = x[:, :, flip_mask].flip(dims=[-1])
        non_flipped_rows = x[:, :, ~flip_mask]
        new_x = torch.empty_like(x)
        new_x[:, :, flip_mask] = flipped_rows
        new_x[:, :, ~flip_mask] = non_flipped_rows
        return new_x

    def forward(self, x):
        # import time
        # starttime = time_synchronized()
        rgb_fea = x[0]  # rgb_fea (tensor): dim:(B, C, num_frames, H, W)
        ir_fea = x[1]  # ir_fea (tensor): dim:(B, C, num_frames, H, W)
        assert rgb_fea.shape[0] == ir_fea.shape[0]

        to_reshape = False
        if len(rgb_fea.shape) == 4:
            to_reshape = True
            bs_f, c, h, w = rgb_fea.shape
            rgb_fea = rgb_fea.view(-1, self.num_frames, c, h, w)
            ir_fea = ir_fea.view(-1, self.num_frames, c, h, w)
        else:
            rgb_fea = rgb_fea.transpose(1, 2)
            ir_fea = ir_fea.transpose(1, 2)

        bs, num_frames, _, h, w = rgb_fea.shape

        mhh_rgb_feas = []
        mhh_ir_feas = []
        for pf_idx, patch_embeding in enumerate(self.patch_embeding_list):
            rgb_fea_patch = patch_embeding(
                rgb_fea.view(bs * num_frames, self.embed_dims, h, w)
            )  # bs*num_frames, embed_dims, patch_h, patch_w
            ir_fea_patch = patch_embeding(
                ir_fea.view(bs * num_frames, self.embed_dims, h, w)
            )  # bs*num_frames, embed_dims, patch_h, patch_w

            if self.order_concat_flatten:
                patch_h, patch_w = rgb_fea_patch.shape[-2:]
                # rgb_fea_patch = rgb_fea_patch.view(bs*num_frames, self.embed_dims_mhh, patch_h, patch_w)
                new_x = self.perform_order_concat_flatten(rgb_fea_patch)
                rgb_fea_patch = (
                    new_x.flatten(-2).transpose(1, 2).reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                )
                rgb_fea_patch = rgb_fea_patch + self.modality_embed[0]
                new_x = self.perform_order_concat_flatten(ir_fea_patch)
                ir_fea_patch = (
                    new_x.flatten(-2).transpose(1, 2).reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                )
                ir_fea_patch = ir_fea_patch + self.modality_embed[1]
                x = torch.stack((rgb_fea_patch, ir_fea_patch), dim=2).view(
                    bs, 2 * num_frames * patch_h * patch_w, self.embed_dims_mhh
                )  # bs, 2*nf*patch_h*patch_w, embed_dims
            else:
                patch_h, patch_w = rgb_fea_patch.shape[-2:]
                rgb_fea_patch = (
                    rgb_fea_patch.reshape(bs, num_frames, self.embed_dims_mhh, patch_h, patch_w)
                    .permute(0, 1, 3, 4, 2)
                    .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                )  # bs, nf*patch_h*patch_w, embed_dims
                rgb_fea_patch = rgb_fea_patch + self.modality_embed[0]
                ir_fea_patch = (
                    ir_fea_patch.reshape(bs, num_frames, self.embed_dims_mhh, patch_h, patch_w)
                    .permute(0, 1, 3, 4, 2)
                    .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                )  # bs, nf*patch_h*patch_w, embed_dims
                ir_fea_patch = ir_fea_patch + self.modality_embed[1]
                x = torch.cat((rgb_fea_patch, ir_fea_patch), dim=1)  # bs, 2*nf*patch_h*patch_w, embed_dims

            if self.if_abs_pos_embed:
                # bs, 2*nf*patch_h*patch_w, embed_dims -> bs, 2*nf, patch_h*patch_w, embed_dims
                x = x.view(bs, 2 * num_frames, patch_h * patch_w, self.embed_dims_mhh)
                x = x + self.pos_embed[pf_idx][:, None, :]
                x = x.view(bs, 2 * num_frames * patch_h * patch_w, self.embed_dims_mhh)
                x = self.pos_drop(x)
            # if self.if_abs_pos_embed:
            #     x = x + self.pos_embed[pf_idx]
            #     x = self.pos_drop(x)

            if self.dual_scan:
                rgb_fea_patch = self.patch_embeding_reversed_list[pf_idx](
                    rgb_fea.view(bs * num_frames, self.embed_dims, h, w)
                )  # bs*num_frames, embed_dims, patch_h, patch_w
                ir_fea_patch = self.patch_embeding_reversed_list[pf_idx](
                    ir_fea.view(bs * num_frames, self.embed_dims, h, w)
                )  # bs*num_frames, embed_dims, patch_h, patch_w
                if self.order_concat_flatten:
                    patch_h, patch_w = rgb_fea_patch.shape[-2:]
                    new_x = self.perform_order_concat_flatten(rgb_fea_patch)
                    rgb_fea_patch = (
                        new_x.flatten(-2)
                        .transpose(1, 2)
                        .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                    )
                    rgb_fea_patch = rgb_fea_patch + self.modality_embed[0]
                    new_x = self.perform_order_concat_flatten(ir_fea_patch)
                    ir_fea_patch = (
                        new_x.flatten(-2)
                        .transpose(1, 2)
                        .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                    )
                    ir_fea_patch = ir_fea_patch + self.modality_embed[1]
                    x_reversed = torch.cat((rgb_fea_patch, ir_fea_patch), dim=1)  # bs, 2*nf*patch_h*patch_w, embed_dims
                else:
                    patch_h, patch_w = rgb_fea_patch.shape[-2:]
                    rgb_fea_patch = (
                        rgb_fea_patch.reshape(bs, num_frames, self.embed_dims_mhh, patch_h, patch_w)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                    )  # bs, nf*patch_h*patch_w, embed_dims
                    rgb_fea_patch = rgb_fea_patch + self.modality_embed[0]
                    ir_fea_patch = (
                        ir_fea_patch.reshape(bs, num_frames, self.embed_dims_mhh, patch_h, patch_w)
                        .permute(0, 1, 3, 4, 2)
                        .reshape(bs, num_frames * patch_h * patch_w, self.embed_dims_mhh)
                    )  # bs, nf*patch_h*patch_w, embed_dims
                    ir_fea_patch = ir_fea_patch + self.modality_embed[1]
                    x_reversed = torch.stack((rgb_fea_patch, ir_fea_patch), dim=2).view(
                        bs, 2 * num_frames * patch_h * patch_w, self.embed_dims_mhh
                    )  # bs, 2*nf*patch_h*patch_w, embed_dims

                if self.if_abs_pos_embed:
                    x = x.view(bs, 2 * num_frames, patch_h * patch_w, self.embed_dims_mhh)
                    x = x + self.pos_embed[pf_idx][:, None, :]
                    x = x.view(bs, 2 * num_frames * patch_h * patch_w, self.embed_dims_mhh)
                    x = self.pos_drop(x)

            residual = None
            # hidden_states = x
            if self.dual_scan:
                hidden_states = torch.cat((x, x_reversed.flip(1)), dim=1)  # bs, 4*nf*patch_h*patch_w, embed_dims_mhh
            else:
                hidden_states = x

            segment_lengths = hidden_states.shape[1] // self.num_frames
            h_states = hidden_states.split(segment_lengths, dim=1)
            h_states_new = []
            residual_new = []
            prev_h_last = None
            lasts = [None] * self.n_layers
            for f_idx, h_state in enumerate(h_states):
                for i in range(self.n_layers):
                    y, h_state, residual = self.attention_list[pf_idx](h_state, residual, layer=i, prev_state=lasts[i])
                if lasts[i] is not None:
                    h_states_new.append(h_state[:, -segment_lengths:, :])
                    residual_new.append(residual[:, -segment_lengths:, :])
                else:
                    h_states_new.append(h_state)
                    residual_new.append(residual)
                lasts[i] = y[:, -1:, :].transpose(1, 2)
            hidden_states = torch.cat(h_states_new, dim=1)
            residual = torch.cat(residual_new, dim=1)

            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            hidden_states = self.norm_f(
                residual.to(dtype=self.norm_f.weight.dtype)
            )  # bs, 4*nf*patch_h*patch_w, embed_dims_mhh
            if self.dual_scan:
                hidden_states = hidden_states.view(bs, 4 * num_frames, patch_h, patch_w, self.embed_dims_mhh)
                hidden_states, hidden_states2 = hidden_states.split(2 * num_frames, dim=1)
                hidden_states = torch.cat(
                    [hidden_states, hidden_states2.flip(1)], dim=-1
                )  # bs, 2*num_frames, patch_h, patch_w, 2*embed_dims_mhh
                hidden_states = hidden_states.view(
                    bs * 2 * num_frames, patch_h, patch_w, 2 * self.embed_dims_mhh
                ).permute(
                    0, 3, 1, 2
                )  # bs*2*num_frames, 2*embed_dims_mhh, patch_h, patch_w
            else:
                hidden_states = hidden_states.view(bs * 2 * num_frames, patch_h, patch_w, self.embed_dims_mhh)
            if self.order_concat_flatten:
                hidden_states = self.perform_order_concat_flatten(
                    hidden_states
                )  # bs*2*num_frames, 2*embed_dims_mhh, patch_h, patch_w
                # hidden_states = torch.stack(new_x, dim=2)
            if self.dual_scan:
                hidden_states = self.up_sample_list[pf_idx](hidden_states).view(
                    bs, 2 * num_frames, 2 * self.embed_dims_mhh, h, w
                )
            else:
                hidden_states = self.up_sample_list[pf_idx](hidden_states).view(
                    bs, 2 * num_frames, self.embed_dims_mhh, h, w
                )
            mhh_rgb_fea = hidden_states[:, 0::2]
            mhh_ir_fea = hidden_states[:, 1::2]
            # mhh_rgb_fea, mhh_ir_fea = hidden_states.split([num_frames, num_frames], dim=1) # bs, num_frames, 2*embed_dims_mhh, h, w
            mhh_rgb_feas.append(mhh_rgb_fea)
            mhh_ir_feas.append(mhh_ir_fea)

        rgb_fea = torch.cat(mhh_rgb_feas, dim=2)  # bs, embed_dims, num_frames, h, w
        ir_fea = torch.cat(mhh_ir_feas, dim=2)  # bs, embed_dims, num_frames, h, w

        # rgb_fea, ir_fea = hidden_states.split([num_frames*h*w, num_frames*h*w], dim=1)
        # rgb_fea = rgb_fea.transpose(1,2).reshape(bs, embed_dims, num_frames, h,w)
        # ir_fea = ir_fea.transpose(1,2).reshape(bs, embed_dims, num_frames, h,w)
        # print("Time: ", time_synchronized()-starttime)

        if to_reshape:
            rgb_fea = rgb_fea.transpose(1, 2).reshape(bs_f, c, h, w)
            ir_fea = ir_fea.transpose(1, 2).reshape(bs_f, c, h, w)
        else:
            rgb_fea = rgb_fea.transpose(1, 2)
            ir_fea = ir_fea.transpose(1, 2)

        return rgb_fea, ir_fea
