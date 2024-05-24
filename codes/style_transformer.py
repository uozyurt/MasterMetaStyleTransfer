import math
from functools import partial
from typing import Any, Callable, List, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from torchvision.ops.stochastic_depth import StochasticDepth


from torchvision.ops.misc import MLP, Permute
from torchvision.ops.stochastic_depth import StochasticDepth
from torchvision.transforms._presets import ImageClassification, InterpolationMode





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
    B, H, W, C = input_q.shape
    # pad feature maps to multiples of window size
    pad_r = (window_size[1] - W % window_size[1]) % window_size[1]
    pad_b = (window_size[0] - H % window_size[0]) % window_size[0]
    x = F.pad(input_q, (0, 0, 0, pad_r, 0, pad_b))
    _, pad_H, pad_W, _ = x.shape

    shift_size = shift_size.copy()
    # If window size is larger than feature size, there is no need to shift window
    if window_size[0] >= pad_H:
        shift_size[0] = 0
    if window_size[1] >= pad_W:
        shift_size[1] = 0

    # cyclic shift
    if sum(shift_size) > 0:
        x = torch.roll(x, shifts=(-shift_size[0], -shift_size[1]), dims=(1, 2))

    # partition windows
    num_windows = (pad_H // window_size[0]) * (pad_W // window_size[1])
    x = x.view(B, pad_H // window_size[0], window_size[0], pad_W // window_size[1], window_size[1], C)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B * num_windows, window_size[0] * window_size[1], C)  # B*nW, Ws*Ws, C

    # multi-head attention
    if logit_scale is not None and qkv_bias is not None:
        qkv_bias = qkv_bias.clone()
        length = qkv_bias.numel() // 3
        qkv_bias[length : 2 * length].zero_()



    ### CHANGE FROM ORIGINAL CODE, START ###

    # qkv = F.linear(x, qkv_weight, qkv_bias)

    # use seperate linear layers for q, k, v to allow cross-attention
    q = F.linear(x, q_weight, q_bias)
    k = F.linear(x, k_weight, k_bias)
    v = F.linear(x, v_weight, v_bias)

    q = q.reshape(x.size(0), x.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    k = k.reshape(x.size(0), x.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)
    v = v.reshape(x.size(0), x.size(1), 1, num_heads, C // num_heads).permute(2, 0, 3, 1, 4)


    ### CHANGE FROM ORIGINAL CODE, END ###



    
    if logit_scale is not None:
        # cosine attention
        attn = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(logit_scale, max=math.log(100.0)).exp()
        attn = attn * logit_scale
    else:
        q = q * (C // num_heads) ** -0.5
        attn = q.matmul(k.transpose(-2, -1))
    # add relative position bias
    attn = attn + relative_position_bias

    if sum(shift_size) > 0:
        # generate attention mask
        attn_mask = x.new_zeros((pad_H, pad_W))
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
        attn = attn.view(x.size(0) // num_windows, num_windows, num_heads, x.size(1), x.size(1))
        attn = attn + attn_mask.unsqueeze(1).unsqueeze(0)
        attn = attn.view(-1, num_heads, x.size(1), x.size(1))

    attn = F.softmax(attn, dim=-1)
    attn = F.dropout(attn, p=attention_dropout)

    x = attn.matmul(v).transpose(1, 2).reshape(x.size(0), x.size(1), C)
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
        window_size: List[int],
        shift_size: List[int],
        num_heads: int,
        qkv_bias: bool = True,
        proj_bias: bool = True,
        attention_dropout: float = 0.0,
        dropout: float = 0.0,
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
        use_MLP_from_outside: bool = False, # ADDED (to able not using the MLP as shared in the style encoder)
    ):
        super().__init__()


        if norm_layer is not None:
            self.norm1 = norm_layer(dim)
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


        self.use_MLP_from_outside = use_MLP_from_outside

        if not self.use_MLP_from_outside:
            self.mlp = MLP(dim, [int(dim * mlp_ratio), dim], activation_layer=nn.GELU, inplace=None, dropout=dropout)


        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.normal_(m.bias, std=1e-6)



    def forward(self,
                input_q: Tensor,
                input_k: Tensor,
                input_v: Tensor,
                MLP_input: Optional[nn.Module] = None,
                calculating_Key_in_encoder: bool = None): # CHANGED FROM ORIGINAL CODE (x -> input_q, input_k, input_v) and ADDED MLP_input
        
        # determine the residual connection input
        if (calculating_Key_in_encoder == True) or (self.use_MLP_from_outside == False):
            x = input_q # if we are calculating the key in the encoder or not using cross-attention, the input_q will be used as the residual connection input
        else:
            x = input_v # if we are calculating Scale or Shift, the input_v will be used as the residual connection input (both Scale and Shift are in V position of the MHA)
        

        if self.use_norm:
            x = x + self.stochastic_depth(self.attn(self.norm1(input_q), self.norm1(input_k), self.norm1(input_v)))
            if not self.use_MLP_from_outside:
                x = x + self.stochastic_depth(self.mlp(self.norm2(x)))
            else:
                assert MLP_input is not None, "If you want to use the MLP from outside, you should provide the MLP_input"
                x = x + self.stochastic_depth(MLP_input(self.norm2(x)))
        else:
            x = x + self.stochastic_depth(self.attn(input_q, input_k, input_v))
            if not self.use_MLP_from_outside:
                x = x + self.stochastic_depth(self.mlp(x))
            else:
                assert MLP_input is not None, "If you want to use the MLP from outside, you should provide the MLP_input"
                x = x + self.stochastic_depth(MLP_input(x))
        return x












if __name__ == "__main__":

    # try the StyleSwinTransformerBlock
    block = StyleSwinTransformerBlock(dim=256,
                                      num_heads=8,
                                      window_size=[8, 8],
                                      shift_size=[4, 4],
                                      dropout=0.0,
                                      attention_dropout=0.0,
                                      qkv_bias=True,
                                      proj_bias=True,
                                      mlp_ratio=4.0,
                                      stochastic_depth_prob=0.1,
                                      norm_layer=None, 
                                      use_MLP_from_outside=False)
    
    x = torch.randn(1, 32, 32, 256)
    out = block(x, x, x)

    print(f"Input shape of the StyleSwinTransformerBlock block: {x.shape}")
    print(f"Output shape of the StyleSwinTransformerBlock block: {out.shape}")
    

