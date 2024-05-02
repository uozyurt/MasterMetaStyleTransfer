import os
from torch import save
from torch.nn import Sequential
from torch.nn.functional import cosine_similarity

# import swin transformer and vgg19 from torchvision
from torchvision.models import swin_transformer, vgg19, VGG19_Weights, vgg19_bn, VGG19_BN_Weights


def download_VGG19_and_create_cutted_model_to_process(absolute_project_path,
                                                      model_save_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                                                      use_vgg19_with_batchnorm=False):
    """
    Loads the VGG19 model from torchvision and saves the model with the last layer being relu 5_1.

    Very Deep Convolutional Networks For Large-Scale Image Recognition: <https://arxiv.org/pdf/1409.1556.pdf>

    Load code: <https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html>
    """

    # get the absolute path of the model save path
    model_save_absolute_path = os.path.join(absolute_project_path, model_save_relative_path)

    if(use_vgg19_with_batchnorm):
          # change the model save path to save the model with batchnorm
          model_save_absolute_path = model_save_absolute_path.replace(".pt", "_bn.pt")

    # if the model is not already saved, download the model and save it
    if not os.path.exists(model_save_absolute_path):
                
        if(use_vgg19_with_batchnorm):
                # get the vgg19 model from torchvision
                vgg19_original = vgg19_bn(weights=VGG19_BN_Weights.IMAGENET1K_V1)

                # get the model features from 0 to 44 (last layer is relu 5_1)
                vgg_19_last_layer_is_relu_5_1_output = Sequential(*list(vgg19_original.features)[0:44])
        else:
                # get the vgg19 model from torchvision
                vgg19_original = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)

                # get the model features from 0 to 30 (last layer is relu 5_1)
                vgg_19_last_layer_is_relu_5_1_output = Sequential(*list(vgg19_original.features)[0:30])
                

        # check if model will be saved in a seperate folder, if not exist, create the folder
        if(len(model_save_relative_path.split("/")) > 1):
            model_save_folder = os.path.join(absolute_project_path, "/".join(model_save_relative_path.split("/")[:-1]))
            if not os.path.exists(model_save_folder):
                os.makedirs(model_save_folder)

        # save the model
        save(vgg_19_last_layer_is_relu_5_1_output, os.path.join(absolute_project_path, model_save_absolute_path))


def download_swin_and_create_cutted_model(absolute_project_path,
                                          model_save_relative_path="weights/swin_B_first_2_stages.pt"):
        """
        Loads the Swin Transformer model from torchvision and saves the model with the first 2 stages.

        Swin Transformer: Hierarchical Vision Transformer using Shifted Windows: <https://arxiv.org/abs/2103.14030>

        Load code: <https://pytorch.org/vision/main/models/swin_transformer.html>
        """

        # get the absolute path of the model save path
        model_save_absolute_path = os.path.join(absolute_project_path, model_save_relative_path)

        # if the model is not already saved, download the model and save it
        if not os.path.exists(model_save_absolute_path):
                # get the swin transformer base model (download the weights from the internet if not already downloaded)
                swin_transformer_base = swin_transformer.swin_b(weights="IMAGENET1K_V1")

                # get the model features from 0 to 4, which correstpondes to the first 2 stages (before 2. patch merging)
                swin_B_first_2_stages = Sequential(*list(swin_transformer_base.features)[:4])

                # check if model will be saved in a seperate folder, if not exist, create the folder
                if(len(model_save_relative_path.split("/")) > 1):
                        model_save_folder = os.path.join(absolute_project_path, "/".join(model_save_relative_path.split("/")[:-1]))
                        if not os.path.exists(model_save_folder):
                                os.makedirs(model_save_folder)

                # save the model
                save(swin_B_first_2_stages, os.path.join(absolute_project_path, model_save_absolute_path))


def get_scaled_self_cosine_distance_map_lower_triangle(A, eps=1e-6):
        """
        This function takes a tensor and calculates the spatial-wise self cosine similarity, scaling the values columns with column sums, leabing the lower triangle of the matrix.
        It is used to calculate the similarity loss in the paper.

        input:
                A: tensor of shape B x C x H x W

        output:
                tensor of shape B x C x H x W
        """

        # flatten the HxW dimensions to get B x C x N from B x C x H x W
        A_flat = A.view(A.size(0), A.size(1), -1)

        # permute the tensor to get B x N x C
        A_flat_permuted = A_flat.permute(0, 2, 1)

        # compute the spatial-wise self cosine similarity (it will be used to calculate D^x_{c,ij} and D^x_{cs,ij} in the paper containing all i and j values)
        self_cos_sim = cosine_similarity(A_flat_permuted.unsqueeze(1), A_flat_permuted.unsqueeze(2), dim=3)

        # sum the columns to get the column sums (it will be used to calculate  SUM FOR ALL K -> [D^x_{c,kj}]  and  SUM FOR ALL K -> [D^x_{cs,kj}]  in the paper, containing all j values)
        self_cos_sim_sum = self_cos_sim.sum(dim=1) + eps # add epsilon to avoid division by zero

        # divide the columns with the column sums (it will be used to calculate   D^x_{c,ij} / SUM FOR ALL K -> [D^x_{c,kj}]   and   D^x_{cs,ij} / SUM FOR ALL K -> [D^x_{cs,kj}]   in the paper containing all i and j values)
        self_cos_sim_scaled = self_cos_sim / self_cos_sim_sum.unsqueeze(1)

        # return the scaled self cosine similarity, leaving the lower triangle of the matrix
        return self_cos_sim_scaled.tril(diagonal=-1)


if(__name__ == "__main__"):


    # get current absolute paths parent directory
    absolute_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # download and save the customized swin transformer model
    download_swin_and_create_cutted_model(absolute_project_path = absolute_project_path,
                                          model_save_relative_path = "weights/swin_B_first_2_stages.pt")

    # download and save the customized vgg19 model
    download_VGG19_and_create_cutted_model_to_process(absolute_project_path = absolute_project_path,
                                                      model_save_relative_path = "weights/vgg_19_last_layer_is_relu_5_1_output.pt")