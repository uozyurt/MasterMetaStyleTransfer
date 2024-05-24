import torch.nn as nn

class Decoder(nn.Module):
    """
    StyleTransferDecoder constructs the image from encoded features using upsampling and convolution layers.
    The design follows the architecture described in "Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization"
    by Huang and Belongie. It consists of multiple upsampling and convolution layers to gradually upscale the feature map
    to the target resolution, interspersed with ReLU activation functions to introduce non-linearities.

    References:
    - Huang, X., & Belongie, S. (2017). Arbitrary Style Transfer in Real-time with Adaptive Instance Normalization.
    In Proceedings of the IEEE International Conference on Computer Vision (pp. 1501-1510).
    """

    def __init__(self,
                 channel_dim: int = 256,
                 initializer: str = "kaiming_normal_",):
        super().__init__()

        assert initializer in ["kaiming_normal_", "kaiming_uniform_", "xavier_normal_", "xavier_uniform_", "orthogonal_", "zeros_"], "Invalid initializer. Please choose one of the following: kaiming_normal_, kaiming_uniform_, xavier_normal_, xavier_uniform_, orthogonal_"

        self.decoder = nn.Sequential(
            nn.Conv2d(channel_dim, channel_dim//2, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(channel_dim//2, channel_dim//2, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(channel_dim//2, channel_dim//2, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(channel_dim//2, channel_dim//2, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(channel_dim//2, channel_dim//4, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(channel_dim//4, channel_dim//4, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(channel_dim//4, channel_dim//8, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),
            
            nn.Upsample(scale_factor=2, mode='nearest'),

            nn.Conv2d(channel_dim//8, channel_dim//8, (3, 3), padding=(1, 1), padding_mode='reflect'),
            nn.ReLU(),

            nn.Conv2d(channel_dim//8, 3, (3, 3), padding=(1, 1), padding_mode='reflect'),
        )


        for m in self.decoder.modules():
            if isinstance(m, nn.Conv2d):
                if initializer == "kaiming_normal_":
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif initializer == "kaiming_uniform_":
                    nn.init.kaiming_uniform_(m.weight, mode='fan_out', nonlinearity='relu')
                elif initializer == "xavier_normal_":
                    nn.init.xavier_normal_(m.weight)
                elif initializer == "xavier_uniform_":
                    nn.init.xavier_uniform_(m.weight)
                elif initializer == "orthogonal_":
                    nn.init.orthogonal_(m.weight)

                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.decoder(x)