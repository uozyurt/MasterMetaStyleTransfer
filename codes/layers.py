import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_



class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
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


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window. This setup allows different updates for 
    each (x, scale, shift) based on the same relationships.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): Dimensions (height, width) of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional): If True, includes a learnable bias to query, key, value matrices.
        qk_scale (float, optional): Custom scaling factor for query-key scores; if None, uses head_dim ** -0.5.
        attn_drop (float, optional): Dropout rate for attention weights.
        proj_drop (float, optional): Dropout rate for output projections.

    Attributes:
        scale (float): Scaling factor for query-key scores.
        relative_position_bias_table (nn.Parameter): Learnable relative positional biases.
        relative_position_index (torch.Tensor): Index map for relative positioning.
        qkv (nn.Linear): Linear layer for generating query, key, and value.
        v_scale (nn.Linear): Linear layer to transform scale input for value computation.
        v_shift (nn.Linear): Linear layer to transform shift input for value computation.
        attn_drop (nn.Dropout): Dropout layer for attention.
        proj_x (nn.Linear): Projection layer for outputs.
        proj_x_drop (nn.Dropout): Dropout layer after projection.
        proj_scale (nn.Linear): Projection layer for scale outputs.
        proj_shift (nn.Linear): Projection layer for shift outputs.
        proj_scale_drop (nn.Dropout): Dropout for scale projection.
        proj_shift_drop (nn.Dropout): Dropout for shift projection.
        softmax (nn.Softmax): Softmax layer for normalization.
    """

    def __init__(self, dim, window_size, num_heads, use_ss=False, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.use_ss = use_ss

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # Layers for queries and keys (shared by K) and separate value layers for K, scale, shift
        self.qkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        if self.use_ss:
            self.v_scale = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_shift = nn.Linear(dim, dim, bias=qkv_bias)

        # Drouput and projection layers
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_x = nn.Linear(dim, dim)
        self.proj_x_drop = nn.Dropout(proj_drop)

        if self.use_ss:
            self.proj_scale = nn.Linear(dim, dim)
            self.proj_shift = nn.Linear(dim, dim)
            self.proj_scale_drop = nn.Dropout(proj_drop)
            self.proj_shift_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, scale=None, shift=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        if self.use_ss:
            v_scale = self.v_scale(scale).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            v_shift = self.v_shift(shift).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj_x(x)
        x = self.proj_drop_x(x)

        if self.use_ss:
            scale = (attn @ v_scale).transpose(1, 2).reshape(B_, N, C)
            scale = self.proj_scale(scale)
            scale = self.proj_scale_drop(scale)

            shift = (attn @ v_shift).transpose(1, 2).reshape(B_, N, C)
            shift = self.proj_shift(shift)
            shift = self.proj_shift_drop(shift)

        return x, scale, shift
