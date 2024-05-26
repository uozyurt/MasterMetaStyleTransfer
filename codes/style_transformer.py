from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.stochastic_depth import StochasticDepth


from torchvision.ops.misc import MLP
from torchvision.ops.stochastic_depth import StochasticDepth
import torchvision.transforms as transforms

from copy import deepcopy



torch.fx.wrap("_patch_merging_pad")


def _get_relative_position_bias(
    relative_position_bias_table: torch.Tensor, relative_position_index: torch.Tensor, window_size: List[int]
) -> torch.Tensor:
    N = window_size[0] * window_size[1]
    relative_position_bias = relative_position_bias_table[relative_position_index]  # type: ignore[index]
    relative_position_bias = relative_position_bias.view(N, N, -1)
    relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)
    return relative_position_bias


torch.fx.wrap("_get_relative_position_bias")





def shifted_window_attention(
    input_q: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_q)
    input_k: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_k)
    input_v: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_v)
    q_weight: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> q_weight)
    k_weight: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> k_weight)
    v_weight: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> v_weight)
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    q_bias: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> q_bias)
    k_bias: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> k_bias)
    v_bias: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> v_bias)
    proj_bias: Optional[Tensor] = None,
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
    B, H, W, C = input_q.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]


    ### CHANGE FROM ORIGINAL CODE, START ###
    
    input_q = F.pad(input_q, (0, 0, 0, pad_r, 0, pad_b))
    input_k = F.pad(input_k, (0, 0, 0, pad_r, 0, pad_b))
    input_v = F.pad(input_v, (0, 0, 0, pad_r, 0, pad_b))
    
    _, pad_H, pad_W, _ = input_q.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        input_q = torch.roll(input_q, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        input_k = torch.roll(input_k, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        input_v = torch.roll(input_v, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    
    input_q = input_q.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    input_k = input_k.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    input_v = input_v.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)

    input_q = input_q.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
    input_k = input_k.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
    input_v = input_v.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    q = F.linear(input_q, q_weight, q_bias)
    k = F.linear(input_k, k_weight, k_bias)
    v = F.linear(input_v, v_weight, v_bias)



    q = q.reshape(q.size(0), q.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    k = k.reshape(k.size(0), k.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    v = v.reshape(v.size(0), v.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)

    ### CHANGE FROM ORIGINAL CODE, END ###

    # scale query
    q = q * (C // num_heads) ** -0.5


    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = input_q.new_zeros((pad_H, pad_W))
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
        attn = attn.view(input_q.size(0) // num_windows, num_windows, num_heads, input_q.size(1), input_q.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, input_q.size(1), input_q.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(input_q.size(0), input_q.size(1), C)
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


torch.fx.wrap("shifted_window_attention")


class ShiftedWindowAttention(nn.Module):
    """
    See :func:`shifted_window_attention`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout


        ### CHANGE FROM ORIGINAL CODE, START ###

        #self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # define seperate linear layers for q, k, v to allow cross-attention
        self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv = nn.Linear(dim, dim, bias=qkv_bias)

        ### CHANGE FROM ORIGINAL CODE, END ###


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

    def forward(self, input_q: Tensor, input_k: Tensor, input_v: Tensor): # CHANGED FROM ORIGINAL CODE (x -> input_q, input_k, input_v)
        """
        Args:
            input_q (Tensor): Tensor with layout of [B, H, W, C]
            input_k (Tensor): Tensor with layout of [B, H, W, C]
            input_v (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()

        
        ### CHANGE FROM ORIGINAL CODE, START ###

        # return shifted_window_attention(
        #     x,
        #     self.qkv.weight,
        #     self.proj.weight,
        #     relative_position_bias,
        #     self.window_size,
        #     self.num_heads,
        #     shift_size=self.shift_size,
        #     attention_dropout=self.attention_dropout,
        #     dropout=self.dropout,
        #     qkv_bias=self.qkv.bias,
        #     proj_bias=self.proj.bias,
        # )

        return shifted_window_attention(
            input_q,
            input_k,
            input_v,
            self.Wq.weight,
            self.Wk.weight,
            self.Wv.weight,
            self.proj.weight,
            relative_position_bias,
            self.window_size,
            self.num_heads,
            shift_size=self.shift_size,
            attention_dropout=self.attention_dropout,
            dropout=self.dropout,
            q_bias=self.Wq.bias,
            k_bias=self.Wk.bias,
            v_bias=self.Wv.bias,
            proj_bias=self.proj.bias,
        )
    

        ### CHANGE FROM ORIGINAL CODE, END ###







class StyleSwinTransformerBlock(nn.Module):
    """
    Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (List[int]): Window size.
        shift_size (List[int]): Shift size for shifted window attention.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        dropout (float): Dropout rate. Default: 0.0.
        attention_dropout (float): Attention dropout rate. Default: 0.0.
        stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        norm_layer (nn.Module): Normalization layer.  Default: nn.LayerNorm.
        attn_layer (nn.Module): Attention layer. Default: ShiftedWindowAttention
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int] = [8, 8],
        shift_size: List[int] = [4, 4],
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        mlp_ratio: float = 4.0,
        stochastic_depth_prob: float = 0.0,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        exclude_MLP_after: bool = False, # ADDED (to able not using the MLP as shared in the style encoder)
    ):
        super().__init__()

        self.exclude_MLP_after = exclude_MLP_after

        if norm_layer is not None:
            self.norm1 = norm_layer(dim)

            if not self.exclude_MLP_after:
                self.norm2 = norm_layer(dim)

            self.use_norm = True
        else:
            self.use_norm = False


        self.attn = ShiftedWindowAttention(
            dim = dim,
            num_heads = num_heads,
            window_size = window_size,
            shift_size = shift_size,
            dropout = dropout,
            attention_dropout = attention_dropout,
            qkv_bias = qkv_bias,
            proj_bias = proj_bias,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")



        if not self.exclude_MLP_after:
            self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=MLP_activation_layer, inplace=None, dropout=dropout)

            for m in self.mlp.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, std=1e-6)



    def forward(self,
                input_q: Tensor,
                input_k: Tensor,
                input_v: Tensor,
                calculating_Key_in_encoder: bool = None): # CHANGED FROM ORIGINAL CODE (x -> input_q, input_k, input_v) and ADDED MLP_input
        
        # determine the residual connection input
        if (calculating_Key_in_encoder == True) or (self.exclude_MLP_after == False):
            x = input_q # if we are calculating the key in the encoder or not using cross-attention, the input_q will be used as the residual connection input
        else:
            x = input_v # if we are calculating Scale or Shift, the input_v will be used as the residual connection input (both Scale and Shift are in V position of the MHA)
        

        if self.use_norm:
            x = x + self.stochastic_depth(self.attn(self.norm1(input_q), self.norm1(input_k), self.norm1(input_v)))
            if not self.exclude_MLP_after:
                x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
            
        else:
            x = x + self.stochastic_depth(self.attn(input_q, input_k, input_v))
            if not self.exclude_MLP_after:
                x = x + self.stochastic_depth(self.mlp(x))
        return x















def shifted_window_attention_for_decoder_last_MHA(
    input_q: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_q)
    input_k: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_k)
    input_v_scale: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_v_scale)
    input_v_shift: Tensor, # CHANGED FROM ORIGINAL CODE (input -> input_v_shift)
    q_weight: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> q_weight)
    k_weight: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> k_weight)
    v_weight_scale: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> v_weight_scale)
    v_weight_shift: Tensor, # CHANGED FROM ORIGINAL CODE (qkv_weight -> v_weight_shift)
    proj_weight: Tensor,
    relative_position_bias: Tensor,
    window_size: List[int],
    num_heads: int,
    shift_size: List[int],
    instance_norm_q: nn.Module,
    instance_norm_k: nn.Module,
    attention_dropout: float = 0.0,
    dropout: float = 0.0,
    q_bias: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> q_bias)
    k_bias: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> k_bias)
    v_bias_scale: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> v_bias_scale)
    v_bias_shift: Optional[Tensor] = None, # CHANGED FROM ORIGINAL CODE (qkv_bias -> v_bias_shift)
    proj_bias: Optional[Tensor] = None,
    use_q_proj: bool = False,
    use_Key_instance_norm_after_linear_transformation: bool = True,
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
    B, H, W, C = input_q.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]


    ### CHANGE FROM ORIGINAL CODE, START ###

    # apply instance normalization to the query
    input_q = instance_norm_q(input_q.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    if not use_Key_instance_norm_after_linear_transformation:
        # apply instance normalization to the key
        input_k = instance_norm_k(input_k.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


    
    input_q = F.pad(input_q, (0, 0, 0, pad_r, 0, pad_b))
    input_k = F.pad(input_k, (0, 0, 0, pad_r, 0, pad_b))
    input_v_scale = F.pad(input_v_scale, (0, 0, 0, pad_r, 0, pad_b))
    input_v_shift = F.pad(input_v_shift, (0, 0, 0, pad_r, 0, pad_b))
    
    _, pad_H, pad_W, _ = input_q.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        input_q = torch.roll(input_q, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        input_k = torch.roll(input_k, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        input_v_scale = torch.roll(input_v_scale, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))
        input_v_shift = torch.roll(input_v_shift, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    
    input_q = input_q.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    input_k = input_k.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    input_v_scale = input_v_scale.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    input_v_shift = input_v_shift.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    
    input_q = input_q.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
    input_k = input_k.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
    input_v_scale = input_v_scale.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C
    input_v_shift = input_v_shift.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if use_q_proj:
        q = F.linear(input_q, q_weight, q_bias)
    else:
        q = input_q
    k = F.linear(input_k, k_weight, k_bias)
    v_scale = F.linear(input_v_scale, v_weight_scale, v_bias_scale)
    v_shift = F.linear(input_v_shift, v_weight_shift, v_bias_shift)


    if use_Key_instance_norm_after_linear_transformation:
        # get the Key into B,C,H,W format
        k = k.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
        k = k.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C).permute(0, 3, 1, 2)

        # apply instance normalization
        k = instance_norm_k(k)

        # get the Key back into B*nW,Ws*Ws,C format
        k = k.permute(0, 2, 3, 1).view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
        k = k.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)


        


    q = q.reshape(q.size(0), q.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    k = k.reshape(k.size(0), k.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    v_scale = v_scale.reshape(v_scale.size(0), v_scale.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)
    v_shift = v_shift.reshape(v_shift.size(0), v_shift.size(1), num_heads, C // num_heads).permute(0, 2, 1, 3)

    ### CHANGE FROM ORIGINAL CODE, END ###

    # scale query
    q = q * (C // num_heads) ** -0.5


    attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = input_q.new_zeros((pad_H, pad_W))
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
        attn = attn.view(input_q.size(0) // num_windows, num_windows, num_heads, input_q.size(1), input_q.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, input_q.size(1), input_q.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)



    # SCALE PART
    x_scale = attn.matmul(v_scale).transpose(1, 2).reshape(input_q.size(0), input_q.size(1), C)
    x_scale = F.linear(x_scale, proj_weight, proj_bias)
    x_scale = F.dropout(x_scale, p=dropout)

    # reverse windows
    x_scale = x_scale.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x_scale = x_scale.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x_scale = torch.roll(x_scale, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x_scale = x_scale[:, :H, :W, :].contiguous()





    x_shift = attn.matmul(v_shift).transpose(1, 2).reshape(input_q.size(0), input_q.size(1), C)
    x_shift = F.linear(x_shift, proj_weight, proj_bias)
    x_shift = F.dropout(x_shift, p=dropout)

    # reverse windows
    x_shift = x_shift.view(B, pad_H // window_size[0], pad_W // window_size[1], window_size[0], window_size[1], C)
    x_shift = x_shift.permute(0, 1, 3, 2, 4, 5).reshape(B, pad_H, pad_W, C)

    # reverse cyclic shift
    if sum(shift_size) > 0:
        x_shift = torch.roll(x_shift, shifts=(shift_size[0], shift_size[1]), dims=(1, 2))

    # unpad features
    x_shift = x_shift[:, :H, :W, :].contiguous()


    
    return x_scale, x_shift




class ShiftedWindowAttention_for_decoder_last_MHA(nn.Module):
    """
    See :func:`shifted_window_attention_for_decoder_last_MHA`.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        window_size: List[int],
        shift_size: List[int],
        instance_norm_q: nn.Module,
        instance_norm_k: nn.Module,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        use_q_proj: bool = False,
        use_Key_instance_norm_after_linear_transformation: bool = True,
    ):
        super().__init__()
        if len(window_size) != 2 or len(shift_size) != 2:
            raise ValueError("window_size and shift_size must be of length 2")
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.dropout = dropout

        self.instance_norm_q = instance_norm_q
        self.instance_norm_k = instance_norm_k

        self.use_q_proj = use_q_proj
        self.use_Key_instance_norm_after_linear_transformation = use_Key_instance_norm_after_linear_transformation

        
        
        # define seperate linear layers for q, k, v to allow cross-attention
        if self.use_q_proj:
            self.Wq = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wk = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv_scale = nn.Linear(dim, dim, bias=qkv_bias)
        self.Wv_shift = nn.Linear(dim, dim, bias=qkv_bias)



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
                input_q: Tensor,
                input_k: Tensor,
                input_v_scale: Tensor,
                input_v_shift: Tensor): # CHANGED FROM ORIGINAL CODE (x -> input_q, input_k, input_v)
        """
        Args:
            input_q (Tensor): Tensor with layout of [B, H, W, C]
            input_k (Tensor): Tensor with layout of [B, H, W, C]
            input_v (Tensor): Tensor with layout of [B, H, W, C]
        Returns:
            Tensor with same layout as input, i.e. [B, H, W, C]
        """
        relative_position_bias = self.get_relative_position_bias()


        if self.use_q_proj:
            return shifted_window_attention_for_decoder_last_MHA(
                input_q,
                input_k,
                input_v_scale,
                input_v_shift,
                self.Wq.weight,
                self.Wk.weight,
                self.Wv_scale.weight,
                self.Wv_shift.weight,
                self.proj.weight,
                relative_position_bias,
                self.window_size,
                self.num_heads,
                self.shift_size,
                self.instance_norm_q,
                self.instance_norm_k,
                self.attention_dropout,
                self.dropout,
                q_bias=self.Wq.bias,
                k_bias=self.Wk.bias,
                v_bias_scale=self.Wv_scale.bias,
                v_bias_shift=self.Wv_shift.bias,
                proj_bias=self.proj.bias,
                use_q_proj=self.use_q_proj,
                use_Key_instance_norm_after_linear_transformation=self.use_Key_instance_norm_after_linear_transformation,
            )
        else:
            return shifted_window_attention_for_decoder_last_MHA(
                input_q,
                input_k,
                input_v_scale,
                input_v_shift,
                None,
                self.Wk.weight,
                self.Wv_scale.weight,
                self.Wv_shift.weight,
                self.proj.weight,
                relative_position_bias,
                self.window_size,
                self.num_heads,
                self.shift_size,
                self.instance_norm_q,
                self.instance_norm_k,
                self.attention_dropout,
                self.dropout,
                q_bias=None,
                k_bias=self.Wk.bias,
                v_bias_scale=self.Wv_scale.bias,
                v_bias_shift=self.Wv_shift.bias,
                proj_bias=self.proj.bias,
                use_q_proj=self.use_q_proj,
                use_Key_instance_norm_after_linear_transformation=self.use_Key_instance_norm_after_linear_transformation
            )
    











class StyleEncoder(nn.Module):
    """
    The StyleEncoder part from the proposed Style Transformer module.
    Args:
        encoder_in_channels (int): Number of input channels.
        encoder_out_channels (int): Number of output channels.
        encoder_num_heads (int): Number of attention heads.
        encoder_window_size (List[int]): Window size.
        encoder_shift_size (List[int]): Shift size for shifted window attention.
        encoder_mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        encoder_dropout (float): Dropout rate. Default: 0.0.
        encoder_attention_dropout (float): Attention dropout rate. Default: 0.0.
        encoder_stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        encoder_norm_layer (nn.Module): Normalization layer.  Default: None. (no normalization layer since the paper states it is harmful in the style encoder)
        encoder_MLP_activation_layer (nn.Module): Activation layer for the MLP. Default: nn.GELU.
        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation: (bool): If True, the processed Key will be used in the Scale and Shift calculation. Default: True.
    """

    def __init__(
        self,
        encoder_dim: int,
        encoder_num_heads: int,
        encoder_window_size: List[int],
        encoder_shift_size: List[int],
        encoder_mlp_ratio: float = 4.0,
        encoder_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        encoder_qkv_bias: bool = True,
        encoder_proj_bias: bool = True,
        encoder_stochastic_depth_prob: float = 0.1,
        encoder_norm_layer: Callable[..., nn.Module] = None,
        encoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation: bool = True,
    ):
        
        
        super().__init__()
        
        self.if_use_processed_Key_in_Scale_and_Shift_calculation = encoder_if_use_processed_Key_in_Scale_and_Shift_calculation

        self.encoder_stochastic_depth_prob = encoder_stochastic_depth_prob
        
        self.stochastic_depth = StochasticDepth(encoder_stochastic_depth_prob, "row")

        self.shared_MHA_without_MLP = StyleSwinTransformerBlock(
            dim = encoder_dim,
            num_heads = encoder_num_heads,
            window_size = encoder_window_size,
            shift_size = encoder_shift_size,
            dropout = encoder_dropout,
            attention_dropout = encoder_attention_dropout,
            qkv_bias = encoder_qkv_bias,
            proj_bias = encoder_proj_bias,
            mlp_ratio = encoder_mlp_ratio,
            stochastic_depth_prob = encoder_stochastic_depth_prob,
            norm_layer = encoder_norm_layer,
            MLP_activation_layer = encoder_MLP_activation_layer,
            exclude_MLP_after = True,
        )
        


        self.encoder_MLP_Key = MLP(encoder_dim, [int(encoder_dim * encoder_mlp_ratio), encoder_dim], activation_layer=encoder_MLP_activation_layer, inplace=None, dropout=encoder_dropout)
        self.encoder_MLP_Scale = MLP(encoder_dim, [int(encoder_dim * encoder_mlp_ratio), encoder_dim], activation_layer=encoder_MLP_activation_layer, inplace=None, dropout=encoder_dropout)
        self.encoder_MLP_Shift = MLP(encoder_dim, [int(encoder_dim * encoder_mlp_ratio), encoder_dim], activation_layer=encoder_MLP_activation_layer, inplace=None, dropout=encoder_dropout)





        for m in [self.encoder_MLP_Key.modules(), self.encoder_MLP_Scale.modules(), self.encoder_MLP_Shift.modules()]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)


    
    def forward(self, Key: Tensor, Scale: Tensor, Shift: Tensor):

        if self.if_use_processed_Key_in_Scale_and_Shift_calculation:
            # calculate the Key first, then use this calculated Key to calculate Scale and Shift
            Key = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Key,
                calculating_Key_in_encoder = True
            )
            Key = Key + self.stochastic_depth(self.encoder_MLP_Key(Key))

            Scale = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Scale,
                calculating_Key_in_encoder = False
            )
            Scale = Scale + self.stochastic_depth(self.encoder_MLP_Scale(Scale))


            Shift = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Shift,
                calculating_Key_in_encoder = False
            )
            Shift = Shift + self.stochastic_depth(self.encoder_MLP_Shift(Shift))
        else:
            # calculate Scale (using unprocessed Key)
            Scale = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Scale,
                calculating_Key_in_encoder = False
            )
            Scale = Scale + self.stochastic_depth(self.encoder_MLP_Scale(Scale))

            # calculate Shift (using unprocessed Key)
            Shift = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Shift,
                calculating_Key_in_encoder = False
            )
            Shift = Shift + self.stochastic_depth(self.encoder_MLP_Shift(Shift))

            # calculate Key lastly
            Key = self.shared_MHA_without_MLP(
                input_q = Key,
                input_k = Key,
                input_v = Key,
                calculating_Key_in_encoder = True
            )
            Key = Key + self.stochastic_depth(self.encoder_MLP_Key(Key))

            
        return Key, Scale, Shift





class StyleDecoder(nn.Module):
    """
    The StyleDecoder part from the proposed Style Transformer module.
    Args:
        decoder_dim (int): Number of input channels.
        decoder_num_heads (int): Number of attention heads.
        decoder_window_size (List[int]): Window size.
        decoder_shift_size (List[int]): Shift size for shifted window attention.
        decoder_mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.0.
        decoder_dropout (float): Dropout rate. Default: 0.0.
        decoder_attention_dropout (float): Attention dropout rate. Default: 0.0.
        decoder_stochastic_depth_prob: (float): Stochastic depth rate. Default: 0.0.
        decoder_norm_layer (nn.Module): Normalization layer.  Default: None. (no normalization layer since the paper states it is harmful in the style encoder)
    """


    def __init__(
        self,
        decoder_dim: int,
        decoder_num_heads: int,
        decoder_window_size: List[int],
        decoder_shift_size: List[int],
        decoder_mlp_ratio: float = 4.0,
        decoder_dropout: float = 0.0,
        decoder_attention_dropout: float = 0.0,
        decoder_qkv_bias: bool = True,
        decoder_proj_bias: bool = True,
        decoder_stochastic_depth_prob: float = 0.1,
        decoder_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        decoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        decoder_use_instance_norm_with_affine: bool = False,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end: bool = False,
        decoder_use_Key_instance_norm_after_linear_transformation: bool = True,
        decoder_exclude_MLP_after_Fcs_self_MHA: bool = False,
    ):
        super().__init__()

        self.decoder_dim = decoder_dim
        self.decoder_num_heads = decoder_num_heads
        self.decoder_mlp_ratio = decoder_mlp_ratio
        self.decoder_MLP_activation_layer = decoder_MLP_activation_layer

        self.decoder_use_instance_norm_with_affine = decoder_use_instance_norm_with_affine
        self.decoder_use_regular_MHA_instead_of_Swin_at_the_end = decoder_use_regular_MHA_instead_of_Swin_at_the_end
        self.decoder_use_Key_instance_norm_after_linear_transformation = decoder_use_Key_instance_norm_after_linear_transformation


        self.MHA_self_attn = StyleSwinTransformerBlock(
            dim = decoder_dim,
            num_heads = decoder_num_heads,
            window_size = decoder_window_size,
            shift_size = decoder_shift_size,
            dropout = decoder_dropout,
            attention_dropout = decoder_attention_dropout,
            qkv_bias = decoder_qkv_bias,
            proj_bias = decoder_proj_bias,
            mlp_ratio = decoder_mlp_ratio,
            stochastic_depth_prob = decoder_stochastic_depth_prob,
            norm_layer = decoder_norm_layer,
            MLP_activation_layer = decoder_MLP_activation_layer,
            exclude_MLP_after = decoder_exclude_MLP_after_Fcs_self_MHA,
        )

        # apply instance normalization
        if self.decoder_use_instance_norm_with_affine:
            self.instance_norm_Query = nn.InstanceNorm2d(decoder_dim, affine=True)
            self.instance_norm_Key = nn.InstanceNorm2d(decoder_dim, affine=True)
        else:
            self.instance_norm = nn.InstanceNorm2d(decoder_dim, affine=False)


        self.stochastic_depth = StochasticDepth(decoder_stochastic_depth_prob, "row")

        self.last_MLP = MLP(decoder_dim, [int(decoder_dim * decoder_mlp_ratio), decoder_dim], activation_layer=decoder_MLP_activation_layer, inplace=None, dropout=decoder_dropout)


        if not self.decoder_use_regular_MHA_instead_of_Swin_at_the_end:
            if self.decoder_use_instance_norm_with_affine:
                self.decoder_MHA_for_sigma_and_mu = ShiftedWindowAttention_for_decoder_last_MHA(
                    dim = decoder_dim,
                    num_heads = decoder_num_heads,
                    window_size = decoder_window_size,
                    shift_size = decoder_shift_size,
                    instance_norm_q=self.instance_norm_Query,
                    instance_norm_k=self.instance_norm_Key,
                    dropout = decoder_dropout,
                    attention_dropout = decoder_attention_dropout,
                    qkv_bias = decoder_qkv_bias,
                    proj_bias = decoder_proj_bias,
                    use_q_proj = False,
                    use_Key_instance_norm_after_linear_transformation = decoder_use_Key_instance_norm_after_linear_transformation,
                )
            else:
                self.decoder_MHA_for_sigma_and_mu = ShiftedWindowAttention_for_decoder_last_MHA(
                    dim = decoder_dim,
                    num_heads = decoder_num_heads,
                    window_size = decoder_window_size,
                    shift_size = decoder_shift_size,
                    instance_norm_q=self.instance_norm,
                    instance_norm_k=self.instance_norm,
                    dropout = decoder_dropout,
                    attention_dropout = decoder_attention_dropout,
                    qkv_bias = decoder_qkv_bias,
                    proj_bias = decoder_proj_bias,
                    use_q_proj = False,
                    use_Key_instance_norm_after_linear_transformation = decoder_use_Key_instance_norm_after_linear_transformation
                )


        else:
            self.linear_transformation_Key = nn.Linear(decoder_dim, decoder_dim)
            self.linear_transformation_Scale = nn.Linear(decoder_dim, decoder_dim)
            self.linear_transformation_Shift = nn.Linear(decoder_dim, decoder_dim)

            self.proj_sigma = nn.Linear(decoder_dim, decoder_dim)
            self.proj_mu = nn.Linear(decoder_dim, decoder_dim)



            for m in self.last_MLP.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.normal_(m.bias, std=1e-6)

        

    def forward(self, Fcs: Tensor, Key: Tensor, Scale: Tensor, Shift: Tensor):

        Query = self.MHA_self_attn(Fcs, Fcs, Fcs)

        if not self.decoder_use_regular_MHA_instead_of_Swin_at_the_end:

            # apply instance normalization
            if self.decoder_use_instance_norm_with_affine:
                Query_IN = self.instance_norm_Query(Query.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                Key = self.instance_norm_Key(Key.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            else:
                Query_IN = self.instance_norm(Query.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
                Key = self.instance_norm(Key.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            
            sigma, mu = self.decoder_MHA_for_sigma_and_mu(Query_IN, Key, Scale, Shift)

            

        else:
            Query_prev_shape = Query.shape # save the shape of Query before reshaping
            Query = Query.view(Query.shape[0], Query.shape[1] * Query.shape[2], Query.shape[3])
            Key = Key.view(Key.shape[0], Key.shape[1] * Key.shape[2], Key.shape[3])
            Scale = Scale.view(Scale.shape[0], Scale.shape[1] * Scale.shape[2], Scale.shape[3])
            Shift = Shift.view(Shift.shape[0], Shift.shape[1] * Shift.shape[2], Shift.shape[3])

            if self.decoder_use_Key_instance_norm_after_linear_transformation:

                Key = self.linear_transformation_Key(Key)
                
                # apply instance normalization
                if self.decoder_use_instance_norm_with_affine:
                    Query_IN = self.instance_norm_Query(Query.permute(0, 2, 1)).permute(0, 2, 1)
                    Key = self.instance_norm_Key(Key.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    Query_IN = self.instance_norm(Query.permute(0, 2, 1)).permute(0, 2, 1)
                    Key = self.instance_norm(Key.permute(0, 2, 1)).permute(0, 2, 1)

                Scale = self.linear_transformation_Scale(Scale)
                Shift = self.linear_transformation_Shift(Shift)
            else:

                # apply instance normalization
                if self.decoder_use_instance_norm_with_affine:
                    Query_IN = self.instance_norm_Query(Query.permute(0, 2, 1)).permute(0, 2, 1)
                    Key = self.instance_norm_Key(Key.permute(0, 2, 1)).permute(0, 2, 1)
                else:
                    Query_IN = self.instance_norm(Query.permute(0, 2, 1)).permute(0, 2, 1)
                    Key = self.instance_norm(Key.permute(0, 2, 1)).permute(0, 2, 1)

                Key = self.linear_transformation_Key(Key)
                Scale = self.linear_transformation_Scale(Scale)
                Shift = self.linear_transformation_Shift(Shift)


            Query = Query.view(Query_prev_shape)
            


            # apply MHA manually

            # scale the query
            Query_IN = Query_IN * (Query_IN.shape[-1] ** -0.5)

            attn = F.softmax(torch.matmul(Query_IN, Key.transpose(-2, -1)), dim=-1)

            sigma = torch.matmul(attn, Scale)
            mu = torch.matmul(attn, Shift)

            # project sigma and mu
            sigma = self.proj_sigma(sigma)
            mu = self.proj_mu(mu)

            # reshape sigma and mu
            sigma = sigma.view(Query_prev_shape)
            mu = mu.view(Query_prev_shape)

        
        # scale and shift the query with sigma and mu
        Query = Query * sigma + mu

        Query = Query + self.stochastic_depth(self.last_MLP(Query))

        # return the Query (new Fcs)
        return Query




class StyleTransformer(nn.Module):
    """
    The proposed Style Transformer module.
    Args:
        encoder_dim (int): Number of input channels for the encoder.
        encoder_num_heads (int): Number of attention heads for the encoder.
        encoder_window_size (List[int]): Window size for the encoder.
        encoder_shift_size (List[int]): Shift size for shifted window attention for the encoder.
        encoder_mlp_ratio (float): Ratio of mlp hidden dim to embedding dim for the encoder. Default: 4.0.
        encoder_dropout (float): Dropout rate for the encoder. Default: 0.0.
        encoder_attention_dropout (float): Attention dropout rate for the encoder. Default: 0.0.
        encoder_stochastic_depth_prob: (float): Stochastic depth rate for the encoder. Default: 0.0.
        encoder_norm_layer (nn.Module): Normalization layer for the encoder.  Default: None. (no normalization layer since the paper states it is harmful in the style encoder)
        encoder_MLP_activation_layer (nn.Module): Activation layer for the MLP in the encoder. Default: nn.GELU.
        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation: (bool): If True, the processed Key will be used in the Scale and Shift calculation. Default: True.
        decoder_dim (int): Number of input channels for the decoder.
        decoder_num_heads (int): Number of attention heads for the decoder.
        decoder_window_size (List[int]): Window size for the decoder.
        decoder_shift_size (List[int]): Shift size for shifted window attention for the decoder.
        decoder_mlp_ratio (float): Ratio of mlp hidden dim to embedding dim for the decoder. Default: 4.0.
        decoder_dropout (float): Dropout rate for the decoder. Default: 0.0.
        decoder_attention_dropout (float): Attention dropout rate for the decoder. Default: 0.0.
        decoder_stochastic_depth_prob: (float): Stochastic depth rate for the decoder. Default: 0.0.
        decoder_norm_layer (nn.Module): Normalization layer for the decoder.  Default: None. (no normalization layer since the paper states it is harmful in the style encoder)
    """

    def __init__(
        self,
        encoder_dim: int,
        decoder_dim: int,
        encoder_num_heads: int,
        decoder_num_heads: int,
        encoder_window_size: List[int],
        decoder_window_size: List[int],
        encoder_shift_size: List[int],
        decoder_shift_size: List[int],
        encoder_mlp_ratio: float = 4.0,
        decoder_mlp_ratio: float = 4.0,
        encoder_dropout: float = 0.0,
        decoder_dropout: float = 0.0,
        encoder_attention_dropout: float = 0.0,
        decoder_attention_dropout: float = 0.0,
        encoder_qkv_bias: bool = True,
        decoder_qkv_bias: bool = True,
        encoder_proj_bias: bool = True,
        decoder_proj_bias: bool = True,
        encoder_stochastic_depth_prob: float = 0.1,
        decoder_stochastic_depth_prob: float = 0.1,
        encoder_norm_layer: Callable[..., nn.Module] = None,
        decoder_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        encoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        decoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation: bool = True,
        decoder_use_instance_norm_with_affine: bool = False,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end: bool = False,
        decoder_use_Key_instance_norm_after_linear_transformation: bool = True,
        decoder_exclude_MLP_after_Fcs_self_MHA: bool = False,
    ):
        super().__init__()

        self.encoder = StyleEncoder(
            encoder_dim = encoder_dim,
            encoder_num_heads = encoder_num_heads,
            encoder_window_size = encoder_window_size,
            encoder_shift_size = encoder_shift_size,
            encoder_mlp_ratio = encoder_mlp_ratio,
            encoder_dropout = encoder_dropout,
            encoder_attention_dropout = encoder_attention_dropout,
            encoder_qkv_bias = encoder_qkv_bias,
            encoder_proj_bias = encoder_proj_bias,
            encoder_stochastic_depth_prob = encoder_stochastic_depth_prob,
            encoder_norm_layer = encoder_norm_layer,
            encoder_MLP_activation_layer = encoder_MLP_activation_layer,
            encoder_if_use_processed_Key_in_Scale_and_Shift_calculation = encoder_if_use_processed_Key_in_Scale_and_Shift_calculation,
        )

        self.decoder = StyleDecoder(
            decoder_dim = decoder_dim,
            decoder_num_heads = decoder_num_heads,
            decoder_window_size = decoder_window_size,
            decoder_shift_size = decoder_shift_size,
            decoder_mlp_ratio = decoder_mlp_ratio,
            decoder_dropout = decoder_dropout,
            decoder_attention_dropout = decoder_attention_dropout,
            decoder_qkv_bias = decoder_qkv_bias,
            decoder_proj_bias = decoder_proj_bias,
            decoder_stochastic_depth_prob = decoder_stochastic_depth_prob,
            decoder_norm_layer = decoder_norm_layer,
            decoder_MLP_activation_layer = decoder_MLP_activation_layer,
            decoder_use_instance_norm_with_affine = decoder_use_instance_norm_with_affine,
            decoder_use_regular_MHA_instead_of_Swin_at_the_end = decoder_use_regular_MHA_instead_of_Swin_at_the_end,
            decoder_use_Key_instance_norm_after_linear_transformation = decoder_use_Key_instance_norm_after_linear_transformation,
            decoder_exclude_MLP_after_Fcs_self_MHA = decoder_exclude_MLP_after_Fcs_self_MHA,
        )

    def forward(self,
                Fc: Tensor,
                Fs: Tensor,
                k: int = 1):
        

        
        Scale = Fs
        Shift = Fs



        
        for i in range(k):
            Fs, Scale, Shift = self.encoder(Fs, Scale, Shift)
            Fc = self.decoder(Fc, Fs, Scale, Shift)



        return Fc
    
                
        

        





if __name__ == "__main__":

    # set seed for reproducibility
    torch.manual_seed(0)


    # # try the StyleSwinTransformerBlock
    # block = StyleSwinTransformerBlock(dim=256,
    #                                   num_heads=8,
    #                                   window_size=[8, 8],
    #                                   shift_size=[4, 4],
    #                                   dropout=0.0,
    #                                   attention_dropout=0.0,
    #                                   qkv_bias=True,
    #                                   proj_bias=True,
    #                                   mlp_ratio=4.0,
    #                                   stochastic_depth_prob=0.1,
    #                                   norm_layer=nn.LayerNorm,
    #                                   exclude_MLP_after=False)
    
    # x = torch.randn(1, 32, 32, 256)
    # out = block(x, x, x)

    # print(f"Input shape of the StyleSwinTransformerBlock block: {x.shape}")
    # print(f"Output shape of the StyleSwinTransformerBlock block: {out.shape}")
    # print("\n")



    # # try the StyleEncoder
    # encoder = StyleEncoder(encoder_dim=256,
    #                        encoder_num_heads=8,
    #                        encoder_window_size=[8, 8],
    #                        encoder_shift_size=[4, 4],
    #                        encoder_mlp_ratio=4.0,
    #                        encoder_dropout=0.0,
    #                        encoder_attention_dropout=0.0,
    #                        encoder_stochastic_depth_prob=0.1,
    #                        encoder_norm_layer=None,
    #                        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation=True)
    

    # Key = torch.randn(1, 32, 32, 256)
    # Scale = torch.randn(1, 32, 32, 256)
    # Shift = torch.randn(1, 32, 32, 256)

    # print(f"Input shape of the StyleEncoder: \nKey: {Key.shape}\nScale: {Scale.shape}\nShift: {Shift.shape}\n")
    
    # Key, Scale, Shift = encoder(Key, Scale, Shift)

    # print(f"Output shape of the StyleEncoder:  \nKey: {Key.shape}\nScale: {Scale.shape}\nShift: {Shift.shape}")
    # print("\n")
    

    # # try the StyleDecoder
    # decoder = StyleDecoder(decoder_dim=256,
    #                        decoder_num_heads=8,
    #                        decoder_window_size=[8, 8],
    #                        decoder_shift_size=[4, 4],
    #                        decoder_mlp_ratio=4.0,
    #                        decoder_dropout=0.0,
    #                        decoder_attention_dropout=0.0,
    #                        decoder_qkv_bias=True,
    #                        decoder_proj_bias=True,
    #                        decoder_stochastic_depth_prob=0.1,
    #                        decoder_norm_layer=nn.LayerNorm,
    #                        decoder_MLP_activation_layer=nn.GELU,
    #                        decoder_use_instance_norm_with_affine=False,
    #                        decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
    #                        decoder_use_Key_instance_norm_after_linear_transformation=True)
    
    # Fcs = torch.randn(1, 32, 32, 256)

    # out = decoder(Fcs, Key, Scale, Shift)

    # print(f"Input shape of the StyleDecoder: \nFcs: {Fcs.shape}\nKey: {Key.shape}\nScale: {Scale.shape}\nShift: {Shift.shape}\n")

    # print(f"Output shape of the StyleDecoder: {out.shape}")











    # try the StyleTransformer
    transformer = StyleTransformer(encoder_dim=256,
                                   decoder_dim=256,
                                   encoder_num_heads=8,
                                   decoder_num_heads=8,
                                   encoder_window_size=[8, 8],
                                   decoder_window_size=[8, 8],
                                   encoder_shift_size=[4, 4],
                                   decoder_shift_size=[4, 4],
                                   encoder_mlp_ratio=4.0,
                                   decoder_mlp_ratio=4.0,
                                   encoder_dropout=0.0,
                                   decoder_dropout=0.0,
                                   encoder_attention_dropout=0.0,
                                   decoder_attention_dropout=0.0,
                                   encoder_qkv_bias=True,
                                   decoder_qkv_bias=True,
                                   encoder_proj_bias=True,
                                   decoder_proj_bias=True,
                                   encoder_stochastic_depth_prob=0.1,
                                   decoder_stochastic_depth_prob=0.1,
                                   encoder_norm_layer=None,
                                   decoder_norm_layer=nn.LayerNorm,
                                   encoder_MLP_activation_layer=nn.GELU,
                                   decoder_MLP_activation_layer=nn.GELU,
                                   encoder_if_use_processed_Key_in_Scale_and_Shift_calculation=True,
                                   decoder_use_instance_norm_with_affine=False,
                                   decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
                                   decoder_use_Key_instance_norm_after_linear_transformation=True,
                                   decoder_exclude_MLP_after_Fcs_self_MHA=False)
    


    import cv2
    import os
    import sys
    # add the project path to the system path
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    sys.path.append(project_absolute_path)

    # import the function to download the swin model and create the cutted model
    from codes.utils import download_swin_and_create_cutted_model


    # get the current relative path for the swin model
    swin_model_relative_path = os.path.join("weights", "Swin_B_first_2_stages.pt")

    # download the model and save it
    download_swin_and_create_cutted_model(absolute_project_path = project_absolute_path,
                                        model_save_relative_path = swin_model_relative_path)
    
    # load the model
    swin_B_first_2_stages = torch.load(os.path.join(project_absolute_path, swin_model_relative_path))

    # set the model to evaluation mode
    swin_B_first_2_stages.eval()



    # define the transform
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with mean and std
    ])
    def apply_transform(image):
        return transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    

    # open the content and style images
    Fc = cv2.imread("codes/images_to_try_loss_function/figure4/figure4_column2_content.png") # content image
    Fs = cv2.imread("codes/images_to_try_loss_function/figure4/figure4_column2_style.png") # style image


    Fc = transform(Fc)
    Fs = transform(Fs)

    Fc = Fc.unsqueeze(0)
    Fs = Fs.unsqueeze(0)


    # get the features from the swin model
    Fc = swin_B_first_2_stages(Fc)
    Fs = swin_B_first_2_stages(Fs)





    # print mean, std and max of Fc and Fs
    print(f"Before StyleTransformer, Fc mean: {Fc.mean():.5f}, Fc std: {Fc.std():.5f}, Fc max: {Fc.max():.5f}")
    print(f"Before StyleTransformer, Fs mean: {Fs.mean():.5f}, Fs std: {Fs.std():.5f}, Fs max: {Fs.max():.5f}")


    out = transformer(Fc, Fs, k=1)

    print(f"Input shape of the StyleTransformer: \nFc: {Fc.shape}\nFs: {Fs.shape}\n")
    print(f"Output shape of the StyleTransformer: {out.shape}")



