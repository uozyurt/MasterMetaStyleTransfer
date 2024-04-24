import torch
import os

# import swin transformer from torchvision
from torchvision.models import swin_transformer



# get the swin transformer base model (download the weights from the internet if not already downloaded)
swin_transformer_base = swin_transformer.swin_b(weights='IMAGENET1K_V1')



# get the model features from 0 to 4, which correstpondes to the first 2 stages (before 2. patch merging)
swin_B_first_2_stages = torch.nn.Sequential(*list(swin_transformer_base.features)[:4])

# save the state_dict of the model
if not os.path.exists('swin_B_first_2_stages.pt'):
        torch.save(swin_B_first_2_stages, 'swin_B_first_2_stages.pt')


