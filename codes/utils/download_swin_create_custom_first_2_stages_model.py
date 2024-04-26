import torch
import os

# import swin transformer from torchvision
from torchvision.models import swin_transformer

def download_swin_and_create_cutted_model(absolute_project_path,
                                          model_save_relative_path="models/swin_B_first_2_stages.pt"):
        """
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
                swin_B_first_2_stages = torch.nn.Sequential(*list(swin_transformer_base.features)[:4])

                # check if model will be saved in a seperate folder, if not exist, create the folder
                if(len(model_save_relative_path.split("/")) > 1):
                        model_save_folder = os.path.join(absolute_project_path, "/".join(model_save_relative_path.split("/")[:-1]))
                        if not os.path.exists(model_save_folder):
                                os.makedirs(model_save_folder)

                # save the model
                torch.save(swin_B_first_2_stages, os.path.join(absolute_project_path, model_save_absolute_path))

if(__name__ == "__main__"):
    # get current absolute paths parent directory
    absolute_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    download_swin_and_create_cutted_model(absolute_project_path = absolute_project_path,
                                                      model_save_relative_path = "models/swin_B_first_2_stages.pt")


