import torch
import torch.nn as nn

from layers import StyleTransformerEncoderBlock, StyleTransformerDecoderBlock


class StyleTransformer(nn.Module):
    """
    A StyleTransformer that iteratively processes style and content through multiple encoder and decoder blocks.

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
    """
    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=4,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.ReLU):
        super().__init__()

        self.encoder = StyleTransformerEncoderBlock(dim, input_resolution, num_heads, window_size,
                                                     shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, act_layer)
        self.decoder = StyleTransformerDecoderBlock(dim, input_resolution, num_heads, window_size,
                                                     shift_size, mlp_ratio, qkv_bias, qk_scale, drop, attn_drop, act_layer)

    def forward(self, content, style, num_steps=4):
        """
        Args:
            content (torch.Tensor): Content features of shape (B, H*W, C).
            style (torch.Tensor): Initial style features of shape (B, H*W, C).
            num_steps (int): Number of transformation steps to iterate through.

        Returns:
            torch.Tensor: The transformed content features after T steps.
        """
        scale = style.clone()  # Initialize scale from style
        shift = style.clone()  # Initialize shift from style

        for _ in range(num_steps):
            style, scale, shift = self.encoder(style, scale, shift)
            content = self.decoder(content, style, scale, shift)

        return content


class Decoder(nn.Module):
    """
    StyleTransferDecoder constructs the image from encoded features using upsampling and convolution layers.
    The design follows the architecture described in "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    by Huang and Belongie. It consists of multiple upsampling and convolution layers to gradually upscale the feature map
    to the target resolution, interspersed with ReLU activation functions to introduce non-linearities.

    Reference:
    Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.
    In Proceedings of the IEEE International Conference on Computer Vision (pp. 1501-1510).
    """

    def __init__(self):
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(128, 128, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(128, 128, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(128, 128, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(128, 64, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(64, 64, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(64, 32, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(32, 32, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(32, 3, (3, 3), padding=(1, 1), padding_mode='reflect'),
        )

    def forward(self, x):
        """
        Forward pass of the StyleTransferDecoder.

        Args:
            x (torch.Tensor): Input tensor of encoded features from the content and style encoding layer.

        Returns:
            torch.Tensor: Output tensor of the stylized image.
        """
        return self.decoder(x)
