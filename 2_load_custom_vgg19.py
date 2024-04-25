import torch
import os
import torch.nn as nn

"""
Very Deep Convolutional Networks For Large-Scale Image Recognition: <https://arxiv.org/pdf/1409.1556.pdf>

Load code: <https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html>
"""


# define a custom VGG19 model to get outputs from relu 2_1, relu 3_1, relu 4_1, relu 5_1
class VGG19_custom(nn.Module):
    def __init__(self, features: nn.Module, init_weights: bool = True) -> None:
        super().__init__()
        self.features = features


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        relu_2_1_output = self.features[:7](x)
        relu_3_1_output = self.features[7:12](relu_2_1_output)
        relu_4_1_output = self.features[12:21](relu_3_1_output)
        relu_5_1_output = self.features[21:30](relu_4_1_output)

        return [relu_2_1_output, relu_3_1_output, relu_4_1_output, relu_5_1_output]

# create a custom VGG19 model with output from relu 2_1, relu 3_1, relu 4_1, relu 5_1
VGG19_custom_output = VGG19_custom(torch.load("models/vgg_19_last_layer_is_relu_5_1_output.pt"))

# try a random input
x = torch.randn(1, 3, 224, 224)

# get the output of the custom VGG19 model
output = VGG19_custom_output(x)

print("Output shapes of the custom VGG19 for loss functions:")
for current_output in output:
    print(current_output.shape)