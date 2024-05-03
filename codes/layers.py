import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple
from typing import Any, Callable, List, Optional
from enum import Enum

import math

from torch import nn, Tensor
import torch.nn.functional as F

from torch.nn import MultiheadAttention
from torchvision.models.swin_transformer import SwinTransformer, SwinTransformerBlock


class V_weight_type(Enum):
    KEY = 0
    SCALE = 1
    SHIFT = 2


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.ReLU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    




# TAKEN FROM PYTHORCH IMPLEMENTATION (torchvision.models.swin_transformer)
# CHANGED TO UTILIZE CROSS ATTENTION
def shifted_window_attention(
    input_Q: Tensor,
    input_K: Tensor,
    input_V: Tensor,
    Q_weight: Tensor,
    K_weight: Tensor,
    V_weight: Tensor,
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    Q_bias: Optional[Tensor] = None,
    K_bias: Optional[Tensor] = None,
    V_bias: Optional[Tensor] = None,
    proj_bias: Optional[Tensor] = None,
    logit_scale: Optional[torch.Tensor] = None,
):
    """
    Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        input (Tensor[N, H, W, C]): The input tensor or 4-dimensions.
        qkv_weight (Tensor[in_dim, out_dim]): The weight tensor of query, key, value.
        proj_weight (Tensor[out_dim, out_dim]): The weight tensor of projection.
        relative_position_bias (Tensor): The learned relative position bias added to attention.
        window_size (List[int]): Window size.
        num_heads (int): Number of attention heads.
        shift_size (List[int]): Shift size for shifted window attention.
        attention_dropout (float): Dropout ratio of attention weight. Default: 0.0.
        dropout (float): Dropout ratio of output. Default: 0.0.
        qkv_bias (Tensor[out_dim], optional): The bias tensor of query, key, value. Default: None.
        proj_bias (Tensor[out_dim], optional): The bias tensor of projection. Default: None.
        logit_scale (Tensor[out_dim], optional): Logit scale of cosine attention for Swin Transformer V2. Default: None.
    Returns:
        Tensor[N, H, W, C]: The output tensor after shifted window attention.
    """
    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    B, H, W, C = input_Q.shape

    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    x_Q = F.pad(input_Q, (0, 0, 0, pad_r, 0, pad_b))
    x_K = F.pad(input_K, (0, 0, 0, pad_r, 0, pad_b))
    x_V = F.pad(input_V, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x_Q.shape


    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        x_Q = torch.roll(x_Q, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        x_K = torch.roll(x_K, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        x_V = torch.roll(x_V, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])


    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    x_Q = x_Q.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x_Q = x_Q.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    x_K = x_K.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x_K = x_K.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    x_V = x_V.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x_V = x_V.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C


    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    q = F.linear(x_Q, Q_weight, Q_bias)
    q = q.reshape(x_Q.size(0), x_Q.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    k = F.linear(x_K, K_weight, K_bias)
    k = k.reshape(x_K.size(0), x_K.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    v = F.linear(x_V, V_weight, V_bias)
    v = v.reshape(x_V.size(0), x_V.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)

    if logit_scale is not None:
        # cosine attention
        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        attn_mask = x_Q.new_zeros((pad_H, pad_W))
        h_slices = ((0, -window_size[0]), (-window_size[0], -shift_size[0]), (-shift_size[0], None))
        w_slices = ((0, -window_size[1]), (-window_size[1], -shift_size[1]), (-shift_size[1], None))
        count = 0
        for h in h_slices:
            for w in w_slices:
                attn_mask[h[0] : h[1], w[0] : w[1]] = count
                count += 1
        attn_mask = attn_mask.view(pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1])
        attn_mask = attn_mask.permute(0, 2, 1, 3).reshape(num_windows, window_size[0] * window_size[1])
        attn_mask = attn_mask.unsqueeze(1) - attn_mask.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        attn = attn.view(x_Q.size(0) // num_windows, num_windows, num_heads, x_Q.size(1), x_Q.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x_Q.size(1), x_Q.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
    x = attn.matmul(v).transpose(1, 2).reshape(x_Q.size(0), x_Q.size(1), C)
    x = F.linear(x, proj_weight, proj_bias)
    x = F.dropout(x, p=dropout)

    # reverse windows
    x = x.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x = x[:, :H, :W, :].contiguous()
    return x




# TAKEN FROM PYTHORCH IMPLEMENTATION (torchvision.models.swin_transformer)
def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias




# TAKEN FROM PYTHORCH IMPLEMENTATION (torchvision.models.swin_transformer)
# CHANGED TO UTILIZE CROSS ATTENTION
class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
        use_different_v_weights: bool = False,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout
        self.use_different_v_weights = use_different_v_weights

        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        self.Q_weight = nn.Linear(dim, dim , bias=qkv_bias)
        self.K_weight = nn.Linear(dim, dim, bias=qkv_bias)
        if not use_different_v_weights:
            self.V_weight = nn.Linear(dim, dim, bias=qkv_bias)
        else:
            self.V_weight_key = nn.Linear(dim, dim, bias=qkv_bias)
            self.V_weight_scale = nn.Linear(dim, dim, bias=qkv_bias)
            self.V_weight_shift = nn.Linear(dim, dim, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim, bias=proj_bias)

        self.define_relative_position_bias_table()
        self.define_relative_position_index()

    def define_relative_position_bias_table(self):
        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * self.window_size[0] - 1) * (2 * self.window_size[1] - 1), self.num_heads)
        )  # 2*Wh-1 * 2*Ww-1, nH
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def define_relative_position_index(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1).flatten()  # Wh*Ww*Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

    def get_relative_position_bias(self) -> torch.Tensor:
        return _get_relative_position_bias(
            self.relative_position_bias_table, self.relative_position_index, self.window_size  # type: ignore[arg-type]
        )

    
    def forward(self,
                Q: Tensor,
                K: Tensor,
                V: Tensor,
                key_type: Optional[V_weight_type] = None,
                ):
        """
        Args:
            x (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()

        # choose the correct V weight
        if not self.use_different_v_weights:
            chosen_V_weight = self.V_weight.weight
            chosen_V_bias = self.V_weight.bias
        else:
            if key_type == V_weight_type.KEY:
                chosen_V_weight = self.V_weight_key.weight
                chosen_V_bias = self.V_weight_key.bias
            elif key_type == V_weight_type.SCALE:
                chosen_V_weight = self.V_weight_scale.weight
                chosen_V_bias = self.V_weight_scale.bias
            elif key_type == V_weight_type.SHIFT:
                chosen_V_weight = self.V_weight_shift.weight
                chosen_V_bias = self.V_weight_shift.bias


        # CHANGED THIS PART FROM ORIGINAL IMPLEMENTATION
        return shifted_window_attention(
            input_Q=Q,
            input_K=K,
            input_V=V,
            Q_weight=self.Q_weight.weight,
            K_weight=self.K_weight.weight,
            V_weight=chosen_V_weight,
            proj_weight=self.proj.weight,
            relative_position_bias=relative_position_bias,
            window_size=self.window_size,
            num_heads=self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            Q_bias=self.Q_weight.bias,
            K_bias=self.K_weight.bias,
            V_bias=chosen_V_bias,
            proj_bias=self.proj.bias,
        )
    







class StyleTransformerEncoderBlock(nn.Module):
    """
    A Style Transformer Encoder Block that uses shifted window based multi-head self-attention (SW-MSA)
    for processing style-related transformations in images. This block supports both cyclic shift and
    standard attention mechanisms, making it suitable for a variety of spatial transformer applications.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Dimensions (height, width) of the input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        shift_size (int): Offset for cyclic shift within the window attention mechanism.
        mlp_ratio (float): Expansion ratio for the MLP block compared to input channels.
        qkv_bias (bool, optional): If True, adds a learnable bias to query, key, value projections.
        qk_scale (float | None, optional): Custom scaling for query-key dot product.
        drop (float, optional): Dropout rate for output projection.
        attn_drop (float, optional): Dropout rate for attention weights.
        act_layer (nn.Module, optional): Type of activation function to use.

    Attributes:
        attn (WindowAttention): The window-based attention mechanism.
        mlp_x (Mlp): MLP for processing the main features.
        mlp_scale (Mlp): MLP for processing the scale adjustments.
        mlp_shift (Mlp): MLP for processing the shift adjustments.
        attn_mask (torch.Tensor): Mask for attention to handle visibility between windows.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 num_heads,
                 window_size=8,
                 shift_size=4,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 proj_bias=True,
                 drop=0.,
                 attn_drop=0.,
                 act_layer=nn.ReLU,
                 use_different_v_weights=False):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.use_different_v_weights = use_different_v_weights

        # Adjust window and shift size based on input dimensions
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        # Attention module
        self.attn = ShiftedWindowAttention(dim,
                                           window_size=to_2tuple(self.window_size),
                                           shift_size=to_2tuple(self.shift_size),
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
                                           proj_bias=proj_bias,
                                           attention_dropout=attn_drop,
                                           use_different_v_weights=self.use_different_v_weights)

        # MLP modules for main, scale, and shift features
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_key = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_scale = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_shift = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)




    def forward(self, key, scale, shift):
        """
        Forward pass of the StyleTransformerEncoderBlock.

        Args:
            x (torch.Tensor): Input features (B, H*W, C).
            scale (torch.Tensor): Scale features (B, H*W, C).
            shift (torch.Tensor): Shift features (B, H*W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated x, scale, and shift after the block processing.
        """
        #TODO: determine if key on scale_provessed and shift_processed should be used after the residual connection or before.

        if self.use_different_v_weights:
            key_processed = self.attn(key, key, key, V_weight_type.KEY)
            scale_processed = self.attn(key, key, scale, V_weight_type.SCALE)
            shift_processed = self.attn(key, key, shift, V_weight_type.SHIFT)
        else:
            key_processed = self.attn(key, key, key)
            scale_processed = self.attn(key, key, scale)
            shift_processed = self.attn(key, key, shift)

        # residual connection
        key = key + key_processed
        scale = scale + scale_processed
        shift = shift + shift_processed

        # MLP processing with residual connection
        key = key + self.mlp_key(key)
        scale = scale + self.mlp_scale(scale)
        shift = shift + self.mlp_shift(shift)


        return key, scale, shift



class StyleTransformerDecoderBlock(nn.Module):


    """
    Implements a decoder block for a Style Transformer, which combines content and style features to generate
    stylized outputs. This block uses a shifted window based multi-head self-attention mechanism for processing
    the content features and a shifted window based multi-head cross-attention mechanism to adaptively adjust
    these features based on style, scale, and shift parameters derived from style features.

    The block leverages two types of attention:
    1. Self-attention on the content features for intra-window interaction.
    2. Cross-attention where the style features affect the transformation of content features.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int, int]): Dimensions (height, width) of the input feature map.
        num_heads (int): Number of attention heads.
        window_size (int): Size of the attention window.
        shift_size (int): Offset for the cyclic shift within the window attention mechanism.
        mlp_ratio (float): Expansion ratio for the MLP block compared to the number of input channels.
        qkv_bias (bool, optional): If set to True, adds a learnable bias to query, key, value projections.
        qk_scale (float | None, optional): Custom scaling factor for query-key dot products in attention mechanisms.
        drop (float, optional): Dropout rate applied to the output of the MLP block.
        attn_drop (float, optional): Dropout rate applied to attention weights.
        act_layer (nn.Module, optional): Activation function used in the MLP block. Defaults to nn.ReLU.

    Attributes:
        attn (WindowAttention): The window-based self-attention mechanism.
        mlp (Mlp): MLP block used after attention operations for transforming the features.
        norm_content (nn.InstanceNorm2d): Instance normalization applied to content features.
        norm_style (nn.InstanceNorm2d): Instance normalization applied to style features.
        attn_mask (torch.Tensor): Attention mask for managing visibility between different windows, particularly useful
                                  when applying the shifted window strategy.
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, proj_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio

        # Adjust window and shift size based on input dimensions
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must be in [0, window_size)"

        # Attention module for content
        self.attn_for_content = ShiftedWindowAttention(dim,
                                           window_size=to_2tuple(self.window_size),
                                           shift_size=to_2tuple(self.shift_size),
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
                                           proj_bias=proj_bias,
                                           attention_dropout=attn_drop,
                                           use_different_v_weights=False)
        
        # Attention module for scale and shift
        self.attn_for_scale_and_shift = ShiftedWindowAttention(dim,
                                           window_size=to_2tuple(self.window_size),
                                           shift_size=to_2tuple(self.shift_size),
                                           num_heads=num_heads,
                                           qkv_bias=qkv_bias,
                                           proj_bias=proj_bias,
                                           attention_dropout=attn_drop,
                                           use_different_v_weights=False)
        


        # MLP module for content
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_content = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # last MLP after scale and shift
        self.mlp_last = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Instance normalization layers for content and style features
        self.norm_content = nn.InstanceNorm2d(dim)
        self.norm_style = nn.InstanceNorm2d(dim)



    def forward(self, content, key, scale, shift):
        """
        Forward pass of the StyleTransformerDecoderBlock.

        Args:
            content (torch.Tensor): Content features of shape (B, H*W, C) where B is the batch size,
                                    H and W are the spatial dimensions, and C is the channel dimension.
            style (torch.Tensor): Style features, shaped similarly to the content features.
            scale (torch.Tensor): Scale factors derived from the style features, used for modifying the content features.
            shift (torch.Tensor): Shift values derived from the style features, used for modifying the content features.

        Returns:
            torch.Tensor: The transformed content features, which have been stylized by the given style, scale, and shift parameters.
        """
        # process content features through self attention with residual connection
        content = content + self.attn_for_content(content, content, content)

        # process content features through MLP with residual connection
        content = content + self.mlp_content(content)

        # apply instance normalization to content and key features by permuting N, W, W, C to N, C, H, W
        content_IN = self.norm_content(content.permute(0, 3, 1, 2))
        key_IN = self.norm_style(key.permute(0, 3, 1, 2))

        # invert permutation
        content_IN = content_IN.permute(0, 2, 3, 1)
        key_IN = key_IN.permute(0, 2, 3, 1)

        # process scale and shift features through attention
        sigma = self.attn_for_scale_and_shift(content_IN, key_IN, scale)
        mu = self.attn_for_scale_and_shift(content_IN, key_IN, shift)


        # apply scale and shift to content features
        content = content * sigma + mu


        # process content features through last MLP with residual connection
        content = content + self.mlp_last(content)

        

        return content
    





if __name__ == "__main__":
    # Test the StyleTransformerEncoderBlock
    block_encoder = StyleTransformerEncoderBlock(dim=256,
                                         input_resolution=(16, 16),
                                         num_heads=8,
                                         window_size=8,
                                         shift_size=4,
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         qk_scale=None,
                                         proj_bias=True,
                                         drop=0.,
                                         attn_drop=0.,
                                         act_layer=nn.ReLU)
                                         
    # Test with B, H, W, C input
    key = torch.randn(4, 32, 32, 256)
    scale = torch.randn(4, 32, 32, 256)
    shift = torch.randn(4, 32, 32, 256)

    key, scale, shift = block_encoder(key, scale, shift)

    print("Encoder output shape:")
    print(key.shape, scale.shape, shift.shape)
    print()

    # Test the StyleTransformerDecoderBlock
    block_decoder = StyleTransformerDecoderBlock(dim=256,
                                         input_resolution=(16, 16),
                                         num_heads=8,
                                         window_size=8,
                                         shift_size=4,
                                         mlp_ratio=4.,
                                         qkv_bias=True,
                                         qk_scale=None,
                                         proj_bias=True,
                                         drop=0.,
                                         attn_drop=0.,
                                         act_layer=nn.ReLU)
    
    # Test with B, H, W, C input
    content = torch.randn(4, 32, 32, 256)

    content = block_decoder(content, key, scale, shift)
    print("Decoder output shape:")
    print(content.shape, scale.shape, shift.shape)