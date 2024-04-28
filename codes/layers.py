import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, to_2tuple



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


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


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
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

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

    def forward(self, q_input, k_input, v_input, scale=None, shift=None, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        q = self.Wq(q_input).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = self.Wk(k_input).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = self.Wv(v_input).reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

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

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
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

        # Attention module
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads, use_ss=True,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # MLP modules for main, scale, and shift features
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp_x = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_scale = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mlp_shift = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # Generate attention mask for shifted window attention if necessary
        if self.shift_size > 0:
            self.attn_mask = self._create_attention_mask(input_resolution)
        else:
            self.attn_mask = None

        self.register_buffer("attn_mask", self.attn_mask)

    def _create_attention_mask(self, input_resolution):
        """
        Creates an attention mask for the shifted window attention mechanism.
        This mask helps in differentiating the shifted window positions during self-attention.

        Args:
            input_resolution (tuple): The dimensions of the input feature map (height, width).

        Returns:
            torch.Tensor: The attention mask with adjustments for shift size.
        """
        H, W = input_resolution
        img_mask = torch.zeros((1, H, W, 1))
        # Define slices for different regions of the image
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        # Window partitioning and attention mask generation
        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0))
        attn_mask = attn_mask.masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

    def forward(self, x, scale, shift):
        """
        Forward pass of the StyleTransformerEncoderBlock.

        Args:
            x (torch.Tensor): Input features (B, H*W, C).
            scale (torch.Tensor): Scale features (B, H*W, C).
            shift (torch.Tensor): Shift features (B, H*W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Updated x, scale, and shift after the block processing.
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        # Convert flat feature maps to spatial maps
        shortcut_x = x
        shortcut_scale = scale
        shortcut_shift = shift

        x = x.view(B, H, W, C)
        scale = scale.view(B, H, W, C)
        shift = shift.view(B, H, W, C)

        # Apply cyclic shift
        if self.shift_size > 0:
            x, scale, shift = self._apply_cyclic_shift(x, scale, shift)

        # Partition windows and process through attention and MLPs
        x, scale, shift = self._process_windows(x, scale, shift, C)

        # Reverse cyclic shift and merge windows
        x, scale, shift = self._reverse_cyclic_shift(x, scale, shift, H, W, B, C)

        # Apply residual connections and MLPs
        x = x + shortcut_x
        scale = scale + shortcut_scale
        shift = shift + shortcut_shift

        x = x + self.mlp_x(x)
        scale = scale + self.mlp_scale(scale)
        shift = shift + self.mlp_shift(shift)

        return x, scale, shift

    def _apply_cyclic_shift(self, x, scale, shift):
        """
        Applies a cyclic shift to the feature maps to enable cross-window connectivity.

        Args:
            x (torch.Tensor): Spatial feature map (B, H, W, C).
            scale (torch.Tensor): Scale feature map (B, H, W, C).
            shift (torch.Tensor): Shift feature map (B, H, W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Shifted feature maps.
        """
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        shifted_scale = torch.roll(scale, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        shifted_shift = torch.roll(shift, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        return shifted_x, shifted_scale, shifted_shift

    def _process_windows(self, x, scale, shift, C):
        """
        Processes the feature maps through window partitioning, attention mechanism,
        and MLPs for each component (main, scale, shift).

        Args:
            x (torch.Tensor): Shifted spatial feature map (B, H, W, C).
            scale (torch.Tensor): Shifted scale feature map (B, H, W, C).
            shift (torch.Tensor): Shifted shift feature map (B, H, W, C).

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Window-processed feature maps.
        """
        # Partition feature maps into windows
        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        scale_windows = window_partition(scale, self.window_size)  # nW*B, window_size, window_size, C
        shift_windows = window_partition(shift, self.window_size)  # nW*B, window_size, window_size, C

        # Flatten and process through attention
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        scale_windows = scale_windows.view(-1, self.window_size * self.window_size, C)
        shift_windows = shift_windows.view(-1, self.window_size * self.window_size, C)

        attn_windows, scale_attn_windows, shift_attn_windows = self.attn(
            x_windows, scale_windows, shift_windows, mask=self.attn_mask)
        
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        scale_attn_windows = scale_attn_windows.view(-1, self.window_size, self.window_size, C)
        shift_attn_windows = shift_attn_windows.view(-1, self.window_size, self.window_size, C)

        return attn_windows, scale_attn_windows, shift_attn_windows

    def _reverse_cyclic_shift(self, attn_windows, scale_attn_windows, shift_attn_windows, H, W, B, C):
        """
        Reverses the cyclic shift applied earlier and merges the windows back into full feature maps.

        Args:
            attn_windows (torch.Tensor): Attended feature windows (nW*B, window_size, window_size, C).
            scale_attn_windows (torch.Tensor): Attended scale windows (nW*B, window_size, window_size, C).
            shift_attn_windows (torch.Tensor): Attended shift windows (nW*B, window_size, window_size, C).
            H (int): Height of the input feature map.
            W (int): Width of the input feature map.
            B (int): Batch size.
            C (int): Number of channels.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Reversed and merged feature maps.
        """
        # Reverse cyclic shift and merge windows back to feature maps
        if self.shift_size > 0:
            shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            shifted_scale = window_reverse(scale_attn_windows, self.window_size, H, W)  # B H' W' C
            scale = torch.roll(shifted_scale, shifts=(self.shift_size, self.shift_size), dims=(1, 2))

            shifted_shift = window_reverse(shift_attn_windows, self.window_size, H, W)  # B H' W' C
            shift = torch.roll(shifted_shift, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = window_reverse(attn_windows, self.window_size, H, W)
            scale = window_reverse(scale_attn_windows, self.window_size, H, W)
            shift = window_reverse(shift_attn_windows, self.window_size, H, W)

        x = x.view(B, H * W, C)
        scale = scale.view(B, H * W, C)
        shift = shift.view(B, H * W, C)

        return x, scale, shift
