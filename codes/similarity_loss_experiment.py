import torch
import torch.nn
from torch.nn.functional import cosine_similarity


torch.manual_seed(0)

# assume we have a tensor of shape B x C x H x W
x = torch.randn(2, 3, 4, 4)

print(x)
print("\n")


# then we compute the cosine similarity mapping between spatial locations (we should end up with a tensor of shape B x (H*W) x (H*W)
x_flat = x.view(x.size(0), x.size(1), -1)
print(x_flat)
print(x_flat.shape)
print("\n")

# permute the tensor to get B x N x C
x_flat_permuted = x_flat.permute(0, 2, 1)
print(x_flat_permuted)
print(x_flat_permuted.shape)
print("\n")

# compute the cosine similarity
cos_sim = cosine_similarity(x_flat_permuted.unsqueeze(1), x_flat_permuted.unsqueeze(2), dim=3)
print(cos_sim)
print(cos_sim.shape)
print("\n")

# sum the columns to get the column sums
cos_sim_sum = cos_sim.sum(dim=1) + 1e-6
print(cos_sim_sum)
print(cos_sim_sum.shape)
print("\n")

# divide the columns with the column sums
cos_sim_scaled = cos_sim / cos_sim_sum.unsqueeze(1)
print(cos_sim_scaled)
print(cos_sim_scaled.shape)
print("\n")

