import torch
import os
import torch.nn as nn
from torchvision import transforms

import sys
# add the project path to the system path
project_absolute_path_from_loss_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_absolute_path_from_loss_py)

# import the function to download the VGG19 model and create the cutted model
from codes.utils import get_scaled_self_cosine_distance_map_lower_triangle, download_VGG19_and_create_cutted_model_to_process

# define the custom VGG19 model using the original VGG19 model as input
class VGG19_custom(nn.Module):
    def __init__(self, features: nn.Module):
        super().__init__()

        # set the features (list of layers of the VGG19 model)
        self.features = features

    # define the forward function
    def forward(self, x):
        # get the output from relu 2_1
        relu_2_1_output = self.features[:7](x)

        # get the output from relu 3_1
        relu_3_1_output = self.features[7:12](relu_2_1_output)

        # get the output from relu 4_1
        relu_4_1_output = self.features[12:21](relu_3_1_output)

        # get the output from relu 5_1
        relu_5_1_output = self.features[21:30](relu_4_1_output)

        # return the outputs as a list
        return [relu_2_1_output, relu_3_1_output, relu_4_1_output, relu_5_1_output]
    

# define the custom VGG19 model using the original VGG19 model with batchnorm as input 
class VGG19_custom_with_batch_norm(nn.Module):
    def __init__(self,  features: nn.Module):
        super().__init__()

        # set the features (list of layers of the VGG19 model)
        self.features = features

    # define the forward function
    def forward(self, x):
        # get the output from relu 2_1
        relu_2_1_output = self.features[:10](x)

        # get the output from relu 3_1
        relu_3_1_output = self.features[10:17](relu_2_1_output)

        # get the output from relu 4_1
        relu_4_1_output = self.features[17:30](relu_3_1_output)

        # get the output from relu 5_1
        relu_5_1_output = self.features[30:43](relu_4_1_output)

        # return the outputs as a list
        return [relu_2_1_output, relu_3_1_output, relu_4_1_output, relu_5_1_output]






# construct the loss class
class custom_loss(nn.Module):
    """
    When this class is initialized, it loads the custom VGG19 model, which is cutted at the last layer of relu 5_1.
    If this cutted model is not saved, it downloads the original VGG19 model and creates the cutted model.
    The class calculates the total loss (content loss + lambda * style loss) for the output image, content image, and style image.
    """
    def __init__(self,
                 project_absolute_path,
                 feature_extractor_model_relative_path=None,
                 use_vgg19_with_batchnorm=False,
                 default_lambda_value=10,
                 distance="euclidian"):
        super().__init__()

        assert distance in ["euclidian", "euclidian_squared"], "distance should be either 'euclidian' or 'euclidian_squared'"


        # if the relative path is not given, set it to the default
        if feature_extractor_model_relative_path is None:
            if use_vgg19_with_batchnorm:
                feature_extractor_model_relative_path = os.path.join("weights", "vgg_19_last_layer_is_relu_5_1_output_bn.pt")
            else:
                feature_extractor_model_relative_path = os.path.join("weights", "vgg_19_last_layer_is_relu_5_1_output.pt")

        # set the lambda value
        self.lambda_value = default_lambda_value


        # set the InstanceNorm2d layers for the content loss part
        self.IN_0 =nn.InstanceNorm2d(128)
        self.IN_1 =nn.InstanceNorm2d(256)
        self.IN_2 =nn.InstanceNorm2d(512)
        self.IN_3 =nn.InstanceNorm2d(512)



        # define the content loss for each term
        if distance == "euclidian_squared":

            self.content_loss_each_term = lambda Fc, Fcs, IN : torch.mean(torch.square(torch.sub(IN(Fc), IN(Fcs))))

        elif distance == "euclidian":

            self.content_loss_each_term = lambda Fc, Fcs, IN : torch.mean(torch.abs(torch.sub(IN(Fc), IN(Fcs))))




        # define the style loss for each term
        if distance == "euclidian_squared":  

            self.style_loss_each_term = lambda Fs, Fcs : torch.mean(torch.square(torch.sub(Fs.mean([2,3]), Fcs.mean([2,3])))) + \
                                                         torch.mean(torch.square(torch.sub(Fs.std([2,3]), Fcs.std([2,3]))))
            
        elif distance == "euclidian":

            self.style_loss_each_term = lambda Fs, Fcs : torch.mean(torch.abs(torch.sub(Fs.mean([2,3]), Fcs.mean([2,3])))) + \
                                                         torch.mean(torch.abs(torch.sub(Fs.std([2,3]), Fcs.std([2,3]))))





        # define the similarity loss for each term
        if distance == "euclidian_squared":
            self.similarity_loss_each_term = lambda normalized_similarity_map_C, normalized_similarity_map_CS : \
                                             torch.mean(torch.square(torch.sub( \
                                                                               get_scaled_self_cosine_distance_map_lower_triangle(normalized_similarity_map_C), \
                                                                               get_scaled_self_cosine_distance_map_lower_triangle(normalized_similarity_map_CS))))
        elif distance == "euclidian":
            self.similarity_loss_each_term = lambda normalized_similarity_map_C, normalized_similarity_map_CS : \
                                             torch.mean(torch.abs(torch.sub( \
                                                                            get_scaled_self_cosine_distance_map_lower_triangle(normalized_similarity_map_C), \
                                                                            get_scaled_self_cosine_distance_map_lower_triangle(normalized_similarity_map_CS))))




        # get the absolute path of the feature extractor model
        feature_extractor_model_path = os.path.join(project_absolute_path, feature_extractor_model_relative_path)


        # check if the VGG19 model is created and saved
        if not os.path.exists(feature_extractor_model_path):

            # create the VGG19 cutted model and save it
            download_VGG19_and_create_cutted_model_to_process(project_absolute_path,
                                                              feature_extractor_model_relative_path,
                                                              use_vgg19_with_batchnorm=use_vgg19_with_batchnorm)

        if use_vgg19_with_batchnorm:
            # load the custom VGG19 model with batchnorm
            self.feature_extractor_model = VGG19_custom_with_batch_norm(torch.load(feature_extractor_model_path))
        else:
            # load the custom VGG19 model without batchnorm
            self.feature_extractor_model = VGG19_custom(torch.load(feature_extractor_model_path))


        # freeze the model
        for param in self.feature_extractor_model.parameters():
            param.requires_grad = False

    # define the forward function
    def forward(self,
                content_image,
                style_image,
                output_image,
                distance="euclidian", # "squared_euclidian" or "euclidian"
                lambda_value=None,
                output_content_and_style_loss=False,
                output_similarity_loss=False):
        """
        Gets the content image, style image, and output image, and returns the total loss (content loss + lambda * style loss)
        All images should be in the exact same shape: [batch_size, 3, 256, 256]
        """
        # check if any lambda value is given explicitly
        if lambda_value is not None:
            lambda_value = self.lambda_value
        return self.get_overall_loss(content_image = content_image,
                                     style_image = style_image,
                                     output_image = output_image,
                                     loss_weight = lambda_value,
                                     output_content_and_style_loss = output_content_and_style_loss,
                                     output_similarity_loss = output_similarity_loss)
    


    # Overall, weighted loss (containin both content and style loss)
    def get_overall_loss(self,
                         content_image,
                         style_image,
                         output_image,
                         loss_weight=None,
                         output_content_and_style_loss=False,
                         output_similarity_loss=False):
        """
        This function calculates the total loss (content loss + lambda * style loss) for the output image.
        It uses the custom VGG19 model to get the outputs from relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers, as it is declared in the paper.
        """
        assert content_image.shape == style_image.shape == output_image.shape, "All images should be in the exact same shape"
        assert content_image.requires_grad == False, "Content image should not require gradient"
        assert style_image.requires_grad == False, "Style image should not require gradient"

        # inputs are in shape: [batch_size, 3, 256, 256]

        # check if lambda value is given
        if loss_weight is None:
            loss_weight = self.lambda_value

        # get the VGG features for content, style, and output images
        VGG_features_content = self.feature_extractor_model(content_image) 
        VGG_features_style = self.feature_extractor_model(style_image)
        VGG_features_output = self.feature_extractor_model(output_image)


        # all above are lists with 4 elements (shapes below are for 256x256 image inputs)
        # first element of each list is the output from relu 2_1 layer,  which is in shape: [batch_size, 128, 128, 128]
        # second element of each list is the output from relu 3_1 layer, which is in shape: [batch_size, 256, 64, 64]
        # third element of each list is the output from relu 4_1 layer,  which is in shape: [batch_size, 512, 32, 32]
        # fourth element of each list is the output from relu 5_1 layer, which is in shape: [batch_size, 512, 16, 16]

        # calculate losses
        content_loss = self.get_content_loss(VGG_features_content,
                                             VGG_features_output)
        
        style_loss = self.get_style_loss(VGG_features_style,
                                         VGG_features_output)

        
        # calculate total loss
        total_loss = content_loss + loss_weight * style_loss

        if output_similarity_loss:
            # calculate similarity loss (passing only relu 4_1 and relu 5_1 layers)
            similarity_loss = self.get_similarity_loss(VGG_features_content, VGG_features_output)

            # if requested, return the content and style loss too
            if output_content_and_style_loss:
                return total_loss, content_loss, style_loss, similarity_loss
            else:
                # return only the total loss
                return total_loss, similarity_loss
            
        else:
            # if requested, return the content and style loss too
            if output_content_and_style_loss:
                return total_loss, content_loss, style_loss
            else:
                # return only the total loss
                return total_loss
    


    # Content Loss
    def get_content_loss(self,
                         VGG_features_content,
                         VGG_features_output):
        """
        calculates the content loss (normalized perceptual loss in <https://arxiv.org/pdf/1603.08155>)

        NOTE: Originally, in the paper cited above, the loss is scaled by W,H,C and euclidian distance is used.
        In the master paper, the loss is ambiguous to be squared distance or euclidian distance.
        Also, it is not explicitly mentioned that the loss is scaled by W,H,C.
        We assumed the loss is squared distance, and scaled by B,W,H,C (by taking mean instead of sum) as it produced closed loss values reported in the paper.

        inputs:
            VGG_features_content: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # calculate content loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1
        content_loss =  self.content_loss_each_term(VGG_features_content[0], VGG_features_output[0], self.IN_0) + \
                        self.content_loss_each_term(VGG_features_content[1], VGG_features_output[1], self.IN_1) + \
                        self.content_loss_each_term(VGG_features_content[2], VGG_features_output[2], self.IN_2) + \
                        self.content_loss_each_term(VGG_features_content[3], VGG_features_output[3], self.IN_3)
        
        return content_loss
    

    # Style Loss
    def get_style_loss(self,
                       VGG_features_style,
                       VGG_features_output):
        """
        calculates the style loss (mean-variance loss in <https://ieeexplore.ieee.org/document/8237429>)

        NOTE: Again, the loss is ambiguous to be squared distance or euclidian distance.
        Also, it is not explicitly mentioned that the loss is scaled by B,W.
        We assumed the loss is squared distance, and scaled by B,W (by taking mean instead of sum) as it produced closed loss values reported in the paper.


        inputs:
            VGG_features_style: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # calculate style loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1
        style_loss =    self.style_loss_each_term(VGG_features_style[0], VGG_features_output[0]) + \
                        self.style_loss_each_term(VGG_features_style[1], VGG_features_output[1]) + \
                        self.style_loss_each_term(VGG_features_style[2], VGG_features_output[2]) + \
                        self.style_loss_each_term(VGG_features_style[3], VGG_features_output[3])
        
        return style_loss



    
    # Similarity Loss
    def get_similarity_loss(self,
                            VGG_features_content_layers,
                            VGG_features_output_layers):
        """
        calculates the similarity loss defined in the paper

        inputs:
            VGG_features_style: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # calculate similarity loss for relu 3_1, relu 4_1 (only these 2 layers used in the paper)
        similarity_loss =   self.similarity_loss_each_term(VGG_features_content_layers[1], VGG_features_content_layers[1]) + \
                            self.similarity_loss_each_term(VGG_features_content_layers[2], VGG_features_content_layers[2])

        return similarity_loss




if __name__ == "__main__":

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    # define the flags to output content and style loss, and similarity loss
    IF_OUTPUT_CONTENT_AND_STYLE_LOSS = True

    # calculating similarity loss is time consuming, so it is disabled by default for the test
    IF_OUTPUT_SIMILARITY_LOSS = False

    USE_NORMALIZATION = True


    # set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")



    # define a function to preprocess the image
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])

    if USE_NORMALIZATION:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])


    def apply_transform(image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return transform(image).unsqueeze(0)
    


    




    # get the absolute path of the project
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))




    # test with figure 9 images from the paper
    content_image_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/content_layer.png"))
    style_image_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/style_layer.png"))
    output_image_1_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_1.png"))
    output_image_3_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_3.png"))
    output_image_5_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_5.png"))

    content_image_figure_9 = apply_transform(content_image_figure_9_raw_image)
    style_image_figure_9 = apply_transform(style_image_figure_9_raw_image)
    output_image_1_figure_9 = apply_transform(output_image_1_figure_9_raw_image)
    output_image_3_figure_9 = apply_transform(output_image_3_figure_9_raw_image)
    output_image_5_figure_9 = apply_transform(output_image_5_figure_9_raw_image)


    # find the content_image_figure_9_raw_image mean and std
    content_image_figure_9_raw_image = cv2.cvtColor(content_image_figure_9_raw_image, cv2.COLOR_BGR2RGB)
    content_image_figure_9_raw_image = content_image_figure_9_raw_image / 255.0
    content_image_figure_9_raw_image = content_image_figure_9_raw_image.reshape(-1, 3)
    content_image_figure_9_raw_image_mean = np.mean(content_image_figure_9_raw_image, axis=0)
    content_image_figure_9_raw_image_std = np.std(content_image_figure_9_raw_image, axis=0)


    


    VGG_models_to_be_tried = ["VGG_without_batchnorm", "VGG_with_batchnorm"]
    # VGG_models_to_be_tried = ["VGG_without_batchnorm"]
    distance_to_be_tried = ["euclidian", "euclidian_squared"]

    for custom_loss_instance_if_batchnorm in VGG_models_to_be_tried:
        for distance in distance_to_be_tried:

            # create an instance of the custom loss class
            if custom_loss_instance_if_batchnorm == "VGG_without_batchnorm":
                custom_loss_instance = custom_loss(project_absolute_path,
                                                feature_extractor_model_relative_path=os.path.join("weights", "vgg_19_last_layer_is_relu_5_1_output.pt"),
                                                use_vgg19_with_batchnorm=False,
                                                default_lambda_value=10,
                                                distance=distance)
            else:
                custom_loss_instance = custom_loss(project_absolute_path,
                                                feature_extractor_model_relative_path=os.path.join("weights", "vgg_19_last_layer_is_relu_5_1_output_bn.pt"),
                                                use_vgg19_with_batchnorm=True,
                                                default_lambda_value=10,
                                                distance=distance)
                                                
                
            # set the model to evaluation mode
            custom_loss_instance.eval()

            # load the model to the device
            custom_loss_instance.to(device)

            
            with torch.no_grad():
                content_image_figure_9 = content_image_figure_9.to(device)
                style_image_figure_9 = style_image_figure_9.to(device)

                # calculate total loss for output_image_1
                output_image_1_figure_9 = output_image_1_figure_9.to(device)
                losses_1_figure_9 = custom_loss_instance(content_image_figure_9,
                                                         style_image_figure_9,
                                                         output_image_1_figure_9,
                                                         output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                         output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                output_image_1_figure_9 = output_image_1_figure_9.cpu()

                # calculate total loss for output_image_3
                output_image_3_figure_9 = output_image_3_figure_9.to(device)
                losses_3_figure_9 = custom_loss_instance(content_image_figure_9,
                                                         style_image_figure_9,
                                                         output_image_3_figure_9,
                                                         output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                         output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                output_image_3_figure_9 = output_image_3_figure_9.cpu()



                # calculate total loss for output_image_5
                output_image_5_figure_9 = output_image_5_figure_9.to(device)
                losses_5_figure_9 = custom_loss_instance(content_image_figure_9,
                                                         style_image_figure_9,
                                                         output_image_5_figure_9,
                                                         output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                         output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                output_image_5_figure_9 = output_image_5_figure_9.cpu()
            



            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_1_figure_9, content_loss_1_figure_9, style_loss_1_figure_9, similarity_loss_1_figure_9 = losses_1_figure_9
                else:
                    total_loss_1_figure_9, content_loss_1_figure_9, style_loss_1_figure_9 = losses_1_figure_9
            else:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_1_figure_9, similarity_loss_1_figure_9 = losses_1_figure_9
                else:
                    total_loss_1_figure_9 = losses_1_figure_9

                    
            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_3_figure_9, content_loss_3_figure_9, style_loss_3_figure_9, similarity_loss_3_figure_9 = losses_3_figure_9
                else:
                    total_loss_3_figure_9, content_loss_3_figure_9, style_loss_3_figure_9 = losses_3_figure_9
            else:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_3_figure_9, similarity_loss_3_figure_9 = losses_3_figure_9
                else:
                    total_loss_3_figure_9 = losses_3_figure_9

                    
            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_5_figure_9, content_loss_5_figure_9, style_loss_5_figure_9, similarity_loss_5_figure_9 = losses_5_figure_9
                else:
                    total_loss_5_figure_9, content_loss_5_figure_9, style_loss_5_figure_9 = losses_5_figure_9
            else:
                if IF_OUTPUT_SIMILARITY_LOSS:
                    total_loss_5_figure_9, similarity_loss_5_figure_9 = losses_5_figure_9
                else:
                    total_loss_5_figure_9 = losses_5_figure_9

            # show the image plot with 3x3 grid
            fig, ax = plt.subplots(3, 3, figsize=(12, 12))

            # for the first row, show the content image, style image, and output image (figure 9, layer 1), adding total loss, content loss, and style loss
            ax[0, 0].imshow(cv2.cvtColor(content_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[0, 0].set_title("Content Image")
            ax[0, 0].axis("off")

            ax[0, 1].imshow(cv2.cvtColor(style_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[0, 1].set_title("Style Image")
            ax[0, 1].axis("off")

            ax[0, 2].imshow(cv2.cvtColor(output_image_1_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[0, 2].set_title("Output Image (Layer 1)")
            ax[0, 2].axis("off")

            # to the right of the output image, add the total loss, content loss, and style loss, adding the title
            ax[0, 2].text(300, 50, f"From figure 9, layer 1 output of the paper", fontsize=12, color="green")
            ax[0, 2].text(300, 100, f"Total Loss:       {total_loss_1_figure_9.item():.3}", fontsize=12, color="red")
            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                ax[0, 2].text(300, 150, f"Content Loss:    {content_loss_1_figure_9.item():.3}", fontsize=12, color="red")
                ax[0, 2].text(300, 200, f"Style Loss:       {style_loss_1_figure_9.item():.3}", fontsize=12, color="red")
                if IF_OUTPUT_SIMILARITY_LOSS:
                    ax[0, 2].text(300, 250, f"Similarity Loss: {similarity_loss_1_figure_9.item():.3}", fontsize=12, color="red")
            elif IF_OUTPUT_SIMILARITY_LOSS:
                ax[0, 2].text(300, 150, f"Similarity Loss: {similarity_loss_1_figure_9.item():.3}", fontsize=12, color="red")



            # for the second row, show the content image, style image, and output image (figure 9, layer 3)
            ax[1, 0].imshow(cv2.cvtColor(content_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[1, 0].set_title("Content Image")
            ax[1, 0].axis("off")

            ax[1, 1].imshow(cv2.cvtColor(style_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[1, 1].set_title("Style Image")
            ax[1, 1].axis("off")

            ax[1, 2].imshow(cv2.cvtColor(output_image_3_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[1, 2].set_title("Output Image (Layer 3)")
            ax[1, 2].axis("off")

            # to the right of the output image, add the total loss, content loss, and style loss, adding the title
            ax[1, 2].text(300, 50, f"From figure 9, layer 3 output of the paper", fontsize=12, color="green")
            ax[1, 2].text(300, 100, f"Total Loss:       {total_loss_3_figure_9.item():.3}", fontsize=12, color="red")
            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                ax[1, 2].text(300, 150, f"Content Loss:    {content_loss_3_figure_9.item():.3}", fontsize=12, color="red")
                ax[1, 2].text(300, 200, f"Style Loss:       {style_loss_3_figure_9.item():.3}", fontsize=12, color="red")
                if IF_OUTPUT_SIMILARITY_LOSS:
                    ax[1, 2].text(300, 250, f"Similarity Loss: {similarity_loss_3_figure_9.item():.3}", fontsize=12, color="red")
            elif IF_OUTPUT_SIMILARITY_LOSS:
                ax[1, 2].text(300, 150, f"Similarity Loss: {similarity_loss_3_figure_9.item():.3}", fontsize=12, color="red")



            # for the third row, show the content image, style image, and output image (figure 9, layer 5)
            ax[2, 0].imshow(cv2.cvtColor(content_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[2, 0].set_title("Content Image")
            ax[2, 0].axis("off")

            ax[2, 1].imshow(cv2.cvtColor(style_image_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[2, 1].set_title("Style Image")
            ax[2, 1].axis("off")

            ax[2, 2].imshow(cv2.cvtColor(output_image_5_figure_9_raw_image, cv2.COLOR_BGR2RGB))
            ax[2, 2].set_title("Output Image (Layer 5)")
            ax[2, 2].axis("off")

            # to the right of the output image, add the total loss, content loss, and style loss
            ax[2, 2].text(300, 50, f"From figure 9, layer 5 output of the paper", fontsize=12, color="green")
            ax[2, 2].text(300, 100, f"Total Loss:       {total_loss_5_figure_9.item():.3}", fontsize=12, color="red")
            if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                ax[2, 2].text(300, 150, f"Content Loss:    {content_loss_5_figure_9.item():.3}", fontsize=12, color="red")
                ax[2, 2].text(300, 200, f"Style Loss:       {style_loss_5_figure_9.item():.3}", fontsize=12, color="red")
                if IF_OUTPUT_SIMILARITY_LOSS:
                    ax[2, 2].text(300, 250, f"Similarity Loss: {similarity_loss_5_figure_9.item():.3}", fontsize=12, color="red")
            elif IF_OUTPUT_SIMILARITY_LOSS:
                ax[2, 2].text(300, 150, f"Similarity Loss: {similarity_loss_5_figure_9.item():.3}", fontsize=12, color="red")

            plt.tight_layout()

            # put a short title
            if custom_loss_instance_if_batchnorm == "VGG_without_batchnorm":
                if distance == "euclidian":
                    fig.suptitle("VGG19 without batchnorm, euclidian distance", fontsize=25)
                elif distance == "euclidian_squared":
                    fig.suptitle("VGG19 without batchnorm, euclidian squared distance", fontsize=25)
            elif custom_loss_instance_if_batchnorm == "VGG_with_batchnorm":
                if distance == "euclidian":
                    fig.suptitle("VGG19 with batchnorm, euclidian distance", fontsize=25)
                elif distance == "euclidian_squared":
                    fig.suptitle("VGG19 with batchnorm, euclidian squared distance", fontsize=25)

            # save the figure
            fig.savefig(os.path.join("codes", "images_to_try_loss_function", f"figure9_losses_{custom_loss_instance_if_batchnorm}_{distance}.png"))

            
            # close the figure
            plt.close(fig)





            # test with figure 4 images from the paper

            for column_index in range(1, 6):
                # define paths
                current_column_content_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_content.png"
                current_column_style_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_style.png"
                current_column_output_AdaAttN_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_output_AdaAttN.png"
                current_column_output_Master_ZS_layer1_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_output_Master_ZS_layer1.png"
                current_column_output_Master_ZS_layer3_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_output_Master_ZS_layer3.png"
                current_column_output_Master_FS_image_relative_path = f"codes/images_to_try_loss_function/figure4/figure4_column{column_index}_output_Master_FS.png"

                # open the images
                content_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_content_image_relative_path))
                style_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_style_image_relative_path))
                output_AdaAttN_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_output_AdaAttN_image_relative_path))
                output_Master_ZS_layer1_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_output_Master_ZS_layer1_image_relative_path))
                output_Master_ZS_layer3_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_output_Master_ZS_layer3_image_relative_path))
                output_Master_FS_image_raw_image = cv2.imread(os.path.join(project_absolute_path, current_column_output_Master_FS_image_relative_path))

                # preprocess the images
                content_image = apply_transform(content_image_raw_image)
                style_image = apply_transform(style_image_raw_image)
                output_AdaAttN_image = apply_transform(output_AdaAttN_image_raw_image)
                output_Master_ZS_layer1_image = apply_transform(output_Master_ZS_layer1_image_raw_image)
                output_Master_ZS_layer3_image = apply_transform(output_Master_ZS_layer3_image_raw_image)
                output_Master_FS_image = apply_transform(output_Master_FS_image_raw_image)

                with torch.no_grad():

                    content_image = content_image.to(device)
                    style_image = style_image.to(device)

                    # calculate total loss for output_AdaAttN_image
                    output_AdaAttN_image = output_AdaAttN_image.to(device)
                    losses_AdaAttN          = custom_loss_instance(content_image,
                                                                   style_image,
                                                                   output_AdaAttN_image,
                                                                   output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                   output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                    output_AdaAttN_image = output_AdaAttN_image.cpu()

                    # calculate total loss for output_Master_ZS_layer1_image
                    output_Master_ZS_layer1_image = output_Master_ZS_layer1_image.to(device)
                    losses_Master_ZS_layer1 = custom_loss_instance(content_image,
                                                                   style_image,
                                                                   output_Master_ZS_layer1_image,
                                                                   output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                   output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                    output_Master_ZS_layer1_image = output_Master_ZS_layer1_image.cpu()

                    # calculate total loss for output_Master_ZS_layer3_image
                    output_Master_ZS_layer3_image = output_Master_ZS_layer3_image.to(device)
                    losses_Master_ZS_layer3 = custom_loss_instance(content_image,
                                                                   style_image,
                                                                   output_Master_ZS_layer3_image,
                                                                   output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                   output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                    output_Master_ZS_layer3_image = output_Master_ZS_layer3_image.cpu()

                    # calculate total loss for output_Master_FS_image
                    output_Master_FS_image = output_Master_FS_image.to(device)
                    losses_Master_FS        = custom_loss_instance(content_image,
                                                                   style_image,
                                                                   output_Master_FS_image,
                                                                   output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                   output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
                    output_Master_FS_image = output_Master_FS_image.cpu()
                
                # show the content and style images at the top row. At the bottom row, show 4 output images, adding the losses next to them
                fig, ax = plt.subplots(5, 2, figsize=(12, 25))

                # show the content image
                ax[0, 0].imshow(cv2.cvtColor(content_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[0, 0].set_title("Content Image")
                ax[0, 0].axis("off")

                # show the style image
                ax[0, 1].imshow(cv2.cvtColor(style_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[0, 1].set_title("Style Image")
                ax[0, 1].axis("off")

                # show the AdaAttN output image
                ax[1, 0].imshow(cv2.cvtColor(output_AdaAttN_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[1, 0].set_title("AdaAttN Output")
                ax[1, 0].axis("off")
                ax[1, 0].text(300, 50, f"AdaAttN Output", fontsize=25, color="green")
                ax[1, 0].text(300, 100, f"Total Loss:       {losses_AdaAttN[0].item():.3}", fontsize=25, color="red")
                if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                    ax[1, 0].text(300, 150, f"Content Loss:    {losses_AdaAttN[1].item():.3}", fontsize=25, color="red")
                    ax[1, 0].text(300, 200, f"Style Loss:       {losses_AdaAttN[2].item():.3}", fontsize=25, color="red")
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[1, 0].text(300, 250, f"Similarity Loss: {losses_AdaAttN[3].item():.3}", fontsize=25, color="red")
                else:
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[1, 0].text(300, 150, f"Similarity Loss: {losses_AdaAttN[1].item():.3}", fontsize=25, color="red")
                
                # make ax[1, 1] empty
                ax[1, 1].axis("off")


                # show the Master_ZS_layer1 output image
                ax[2, 0].imshow(cv2.cvtColor(output_Master_ZS_layer1_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[2, 0].set_title("Master ZS Layer 1 Output")
                ax[2, 0].axis("off")
                ax[2, 0].text(300, 50, f"Master ZS Layer 1 Output", fontsize=25, color="green")
                ax[2, 0].text(300, 100, f"Total Loss:       {losses_Master_ZS_layer1[0].item():.3}", fontsize=25, color="red")
                if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                    ax[2, 0].text(300, 150, f"Content Loss:    {losses_Master_ZS_layer1[1].item():.3}", fontsize=25, color="red")
                    ax[2, 0].text(300, 200, f"Style Loss:       {losses_Master_ZS_layer1[2].item():.3}", fontsize=25, color="red")
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[2, 0].text(300, 250, f"Similarity Loss: {losses_Master_ZS_layer1[3].item():.3}", fontsize=25, color="red")
                else:
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[2, 0].text(300, 150, f"Similarity Loss: {losses_Master_ZS_layer1[1].item():.3}", fontsize=25, color="red")
                
                # make ax[2, 1] empty
                ax[2, 1].axis("off")


                # show the Master_ZS_layer3 output image
                ax[3, 0].imshow(cv2.cvtColor(output_Master_ZS_layer3_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[3, 0].set_title("Master ZS Layer 3 Output")
                ax[3, 0].axis("off")
                ax[3, 0].text(300, 50, f"Master ZS Layer 3 Output", fontsize=25, color="green")
                ax[3, 0].text(300, 100, f"Total Loss:       {losses_Master_ZS_layer3[0].item():.3}", fontsize=25, color="red")
                if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                    ax[3, 0].text(300, 150, f"Content Loss:    {losses_Master_ZS_layer3[1].item():.3}", fontsize=25, color="red")
                    ax[3, 0].text(300, 200, f"Style Loss:       {losses_Master_ZS_layer3[2].item():.3}", fontsize=25, color="red")
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[3, 0].text(300, 250, f"Similarity Loss: {losses_Master_ZS_layer3[3].item():.3}", fontsize=25, color="red")
                else:
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[3, 0].text(300, 150, f"Similarity Loss: {losses_Master_ZS_layer3[1].item():.3}", fontsize=25, color="red")
                
                # make ax[3, 1] empty
                ax[3, 1].axis("off")


                # show the Master_FS output image
                ax[4, 0].imshow(cv2.cvtColor(output_Master_FS_image_raw_image, cv2.COLOR_BGR2RGB))
                ax[4, 0].set_title("Master FS Output")
                ax[4, 0].axis("off")
                ax[4, 0].text(300, 50, f"Master FS Output", fontsize=25, color="green")
                ax[4, 0].text(300, 100, f"Total Loss:       {losses_Master_FS[0].item():.3}", fontsize=25, color="red")
                if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
                    ax[4, 0].text(300, 150, f"Content Loss:    {losses_Master_FS[1].item():.3}", fontsize=25, color="red")
                    ax[4, 0].text(300, 200, f"Style Loss:       {losses_Master_FS[2].item():.3}", fontsize=25, color="red")
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[4, 0].text(300, 250, f"Similarity Loss: {losses_Master_FS[3].item():.3}", fontsize=25, color="red")
                else:
                    if IF_OUTPUT_SIMILARITY_LOSS:
                        ax[4, 0].text(300, 150, f"Similarity Loss: {losses_Master_FS[1].item():.3}", fontsize=25, color="red")
                
                # make ax[4, 1] empty
                ax[4, 1].axis("off")




                plt.tight_layout()

                # put a short title
                if custom_loss_instance_if_batchnorm == "VGG_without_batchnorm":
                    if distance == "euclidian":
                        fig.suptitle("VGG19 without batchnorm, euclidian distance", fontsize=25)
                    elif distance == "euclidian_squared":
                        fig.suptitle("VGG19 without batchnorm, euclidian squared distance", fontsize=25)
                elif custom_loss_instance_if_batchnorm == "VGG_with_batchnorm":
                    if distance == "euclidian":
                        fig.suptitle("VGG19 with batchnorm, euclidian distance", fontsize=25)
                    elif distance == "euclidian_squared":
                        fig.suptitle("VGG19 with batchnorm, euclidian squared distance", fontsize=25)
                
                # save the figure
                fig.savefig(os.path.join("codes", "images_to_try_loss_function", f"figure4_losses_column{column_index}_{custom_loss_instance_if_batchnorm}_{distance}.png"))

                # close the figure
                plt.close(fig)

                # print the losses
                print(f"Figure-4 Column-{column_index} losses:")
                print(f"AdaAttN:           {[round(x.item(), 3) for x in losses_AdaAttN]}")
                print(f"Master ZS Layer 1: {[round(x.item(), 3) for x in losses_Master_ZS_layer1]}")
                print(f"Master ZS Layer 3: {[round(x.item(), 3) for x in losses_Master_ZS_layer3]}")
                print(f"Master FS:         {[round(x.item(), 3) for x in losses_Master_FS]}")
                print("\n\n\n")
