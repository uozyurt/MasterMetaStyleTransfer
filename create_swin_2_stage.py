import torch




# load the swin_B_first_2_stages backbone
swin_B_first_2_stages = torch.load('swin_B_first_2_stages.pt')

# try a random input
x = torch.randn(1, 3, 224, 224)

# get the output of the first 2 stages
output = swin_B_first_2_stages(x)

print(f"output shape of the swin transformer backbone (with only first 2 stages):\n{output.shape}\n")