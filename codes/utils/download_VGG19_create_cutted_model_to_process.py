import torch
import torchvision
import os
import torch.nn as nn
import sys

def download_VGG19_and_create_cutted_model_to_process(absolute_project_path,
                                                      model_save_relative_path="models/vgg_19_last_layer_is_relu_5_1_output.pt"):
    """
    # load VGG19 model from torchvision
    Very Deep Convolutional Networks For Large-Scale Image Recognition: <https://arxiv.org/pdf/1409.1556.pdf>

    Load code: <https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html>
    """

    # get the absolute path of the model save path
    model_save_absolute_path = os.path.join(absolute_project_path, model_save_relative_path)

    # if the model is not already saved, download the model and save it
    if not os.path.exists(model_save_absolute_path):
        # get the vgg19 model from torchvision
        vgg19 = torchvision.models.vgg19(pretrained=True)

        # get the model features from 0 to 30 (last layer is relu 5_1)
        vgg_19_last_layer_is_relu_5_1_output = torch.nn.Sequential(*list(vgg19.features)[0:30])

        # check if model will be saved in a seperate folder, if not exist, create the folder
        if(len(model_save_relative_path.split("/")) > 1):
            model_save_folder = os.path.join(absolute_project_path, "/".join(model_save_relative_path.split("/")[:-1]))
            if not os.path.exists(model_save_folder):
                os.makedirs(model_save_folder)

        # save the model
        torch.save(vgg_19_last_layer_is_relu_5_1_output, os.path.join(absolute_project_path, model_save_absolute_path))

if __name__ == "__main__":
    # get current absolute paths parent directory
    absolute_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))

    download_VGG19_and_create_cutted_model_to_process(absolute_project_path = absolute_project_path,
                                                      model_save_relative_path = "models/vgg_19_last_layer_is_relu_5_1_output.pt")