from torch.nn import Sequential
from torch import save
import os

# import swin transformer from torchvision
from torchvision.models import swin_transformer, vgg19


def download_VGG19_and_create_cutted_model_to_process(absolute_project_path,
                                                      model_save_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt"):
    """
    Loads the VGG19 model from torchvision and saves the model with the last layer being relu 5_1.

    Very Deep Convolutional Networks For Large-Scale Image Recognition: <https://arxiv.org/pdf/1409.1556.pdf>

    Load code: <https://pytorch.org/vision/0.8/_modules/torchvision/models/vgg.html>
    """

    # get the absolute path of the model save path
    model_save_absolute_path = os.path.join(absolute_project_path, model_save_relative_path)

    # if the model is not already saved, download the model and save it
    if not os.path.exists(model_save_absolute_path):
        # get the vgg19 model from torchvision
        vgg19 = vgg19(pretrained=True)

        # get the model features from 0 to 30 (last layer is relu 5_1)
        vgg_19_last_layer_is_relu_5_1_output = Sequential(*list(vgg19.features)[0:30])

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



if(__name__ == "__main__"):


    # get current absolute paths parent directory
    absolute_project_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # download and save the customized swin transformer model
    download_swin_and_create_cutted_model(absolute_project_path = absolute_project_path,
                                          model_save_relative_path = "weights/swin_B_first_2_stages.pt")

    # download and save the customized vgg19 model
    download_VGG19_and_create_cutted_model_to_process(absolute_project_path = absolute_project_path,
                                                      model_save_relative_path = "weights/vgg_19_last_layer_is_relu_5_1_output.pt")