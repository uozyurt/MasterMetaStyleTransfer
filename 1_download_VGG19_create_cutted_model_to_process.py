import torch
import torchvision
import os
import torch.nn as nn

# load VGG19 model from torchvision
"""
Very Deep Convolutional Networks For Large-Scale Image Recognition: <https://arxiv.org/pdf/1409.1556.pdf>

Load code: <https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html>
"""
vgg19 = torchvision.models.vgg19(pretrained=True)

vgg_19_last_layer_is_relu_5_1_output = torch.nn.Sequential(*list(vgg19.features)[0:30])


# save the custom VGG19 model
if not os.path.exists("models/vgg_19_last_layer_is_relu_5_1_output.pt"):
    torch.save(vgg_19_last_layer_is_relu_5_1_output, "models/vgg_19_last_layer_is_relu_5_1_output.pt")
