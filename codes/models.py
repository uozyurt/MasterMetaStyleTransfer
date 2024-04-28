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


class StyleDecoder(nn.Module):
    """
    StyleTransferDecoder constructs the image from encoded features using upsampling and convolution layers.
    The design follows the architecture described in "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    by Huang and Belongie. It consists of multiple upsampling and convolution layers to gradually upscale the feature map
    to the target resolution, interspersed with ReLU activation functions to introduce non-linearities.

    References:
    - Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.
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


class SwinEncoder(torch.nn.Module):
    def __init__(self, model_path=None, freeze_params=False):
        """
        Initializes the SwinEncoder object which uses the first two stages of the Swin Transformer.

        Args:
        model_path (str, optional): Path where the Swin model is saved or should be saved. Defaults to '../ckpt/swin_B_first_2_stages.pt'.
        freeze_params (bool): If True, the parameters of the model will be frozen.

        References:
        - Liu, Ze, et al. "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows." arXiv preprint arXiv:2103.14030 (2021). https://arxiv.org/abs/2103.14030
        - Official PyTorch Vision documentation for Swin Transformers: https://pytorch.org/vision/stable/models.html#torchvision.models.swin_transformer
        """
        super().__init__()
        if model_path is None:
            # Resolve the default path to be '../ckpt/swin_B_first_2_stages.pt' relative to the script's location
            script_directory = os.path.dirname(__file__)
            default_directory = os.path.abspath(os.path.join(script_directory, '..', 'ckpt'))
            model_path = os.path.join(default_directory, 'swin_B_first_2_stages.pt')
        
        self.model_path = model_path
        self.model = self.load_or_download_model()
        if freeze_params:
            self.freeze_parameters()

    def load_or_download_model(self):
        """
        This method checks if the model exists at the specified path and loads it.
        If the model does not exist, it downloads the Swin Transformer base model, extracts
        the first two stages, and saves it.
        
        Returns:
        torch.nn.Module: The loaded or downloaded Swin Transformer first two stages.
        """
        if not os.path.exists(self.model_path):
            # Get the Swin Transformer base model
            swin_transformer_base = swin_transformer.swin_b(weights="IMAGENET1K_V1")
            # Extract the first two stages of the model
            swin_first_2_stages = torch.nn.Sequential(*list(swin_transformer_base.features)[:4])
            
            # Ensure the directory exists
            model_dir = os.path.dirname(self.model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                
            # Save the modified model
            torch.save(swin_first_2_stages, self.model_path)
        
        # Load the model
        return torch.load(self.model_path)

    def freeze_parameters(self):
        """
        This method sets the requires_grad attribute of all parameters in the model to False,
        effectively freezing the model to prevent any of its weights from being updated.
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass through the first two stages of the Swin Transformer.

        Args:
        x (torch.Tensor): Input tensor to be passed through the model.

        Returns:
        torch.Tensor: Output tensor from the model.
        """
        return self.model(x)
