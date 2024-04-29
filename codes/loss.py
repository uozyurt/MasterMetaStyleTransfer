import torch
import os
import torch.nn as nn
import sys

# add the project path to the system path
project_absolute_path_from_loss_py = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# add the project path to the system path
sys.path.append(project_absolute_path_from_loss_py)

# import the function to download the VGG19 model and create the cutted model
from codes.utils import get_scaled_self_cosine_distance_map

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


# construct the loss class
class custom_loss(nn.Module):
    """
    When this class is initialized, it loads the custom VGG19 model, which is cutted at the last layer of relu 5_1.
    If this cutted model is not saved, it downloads the original VGG19 model and creates the cutted model.
    The class calculates the total loss (content loss + lambda * style loss) for the output image, content image, and style image.
    """
    def __init__(self,
                 project_absolute_path,
                 feature_extractor_model_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                 default_lambda_value=10):
        super().__init__()

        # set the lambda value
        self.lambda_value = default_lambda_value

        # get the absolute path of the feature extractor model
        feature_extractor_model_path = os.path.join(project_absolute_path, feature_extractor_model_relative_path)

        # check if the VGG19 model is created and saved
        if not os.path.exists(feature_extractor_model_path):
            # add the project path to the system path
            import sys
            sys.path.append(project_absolute_path)

            # import the function to download the VGG19 model and create the cutted model
            from codes.utils import download_VGG19_and_create_cutted_model_to_process

            # create the VGG19 cutted model and save it
            download_VGG19_and_create_cutted_model_to_process(project_absolute_path,
                                                              feature_extractor_model_relative_path)

        # load the custom VGG19 model
        self.feature_extractor_model = VGG19_custom(torch.load(feature_extractor_model_path))


        # set the model to evaluation mode
        self.feature_extractor_model.eval()

        # freeze the model
        for param in self.feature_extractor_model.parameters():
            param.requires_grad = False

    # define the forward function
    def forward(self,
                content_image,
                style_image,
                output_image,
                output_content_and_style_loss=False,
                output_similarity_loss=False):
        """
        Gets the content image, style image, and output image, and returns the total loss (content loss + lambda * style loss)
        All images should be in the exact same shape: [batch_size, 3, 256, 256]
        """
        return self.get_overall_loss(content_image = content_image,
                                     style_image = style_image,
                                     output_image = output_image,
                                     loss_weight = self.lambda_value,
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
        # inputs are in shape: [batch_size, 3, 256, 256]

        # check if lambda value is given
        if loss_weight is None:
            loss_weight = self.lambda_value

        # get the VGG features for content, style, and output images
        VGG_features_content = self.feature_extractor_model(content_image) 
        VGG_features_style = self.feature_extractor_model(style_image)
        VGG_features_output = self.feature_extractor_model(output_image)

        # all above are lists with 4 elements
        # first element of each list is the output from relu 2_1 layer,  which is in shape: [batch_size, 128, 128, 128]
        # second element of each list is the output from relu 3_1 layer, which is in shape: [batch_size, 256, 64, 64]
        # third element of each list is the output from relu 4_1 layer,  which is in shape: [batch_size, 512, 32, 32]
        # fourth element of each list is the output from relu 5_1 layer, which is in shape: [batch_size, 512, 16, 16]

        # calculate losses
        content_loss = self.get_content_loss(VGG_features_content, VGG_features_output)
        style_loss = self.get_style_loss(VGG_features_style, VGG_features_output)

        
        # calculate total loss
        total_loss = content_loss + loss_weight * style_loss

        if output_similarity_loss:
            # calculate similarity loss (passing only relu 4_1 and relu 5_1 layers)
            similarity_loss = self.get_similarity_loss(VGG_features_content[-2:], VGG_features_output[-2:])

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
    def get_content_loss(self, VGG_features_content, VGG_features_output):
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

        # define content loss for each term
        content_loss_each_term = lambda A1, A2, instance_norm: torch.mean(torch.square(torch.sub(instance_norm(A1), instance_norm(A2))))

        # get the shapes of the tensors
        features_shape = VGG_features_content[0].shape

        # calculate content loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1 (also scaled by W,H,C, as in the mentioned paper)
        content_loss =  content_loss_each_term(VGG_features_content[0], VGG_features_output[0], nn.InstanceNorm2d(128)) + \
                        content_loss_each_term(VGG_features_content[1], VGG_features_output[1], nn.InstanceNorm2d(256)) + \
                        content_loss_each_term(VGG_features_content[2], VGG_features_output[2], nn.InstanceNorm2d(512)) + \
                        content_loss_each_term(VGG_features_content[3], VGG_features_output[3], nn.InstanceNorm2d(512))
        

        return content_loss

    # Style Loss
    def get_style_loss(self, VGG_features_style, VGG_features_output):
        """
        calculates the style loss (mean-variance loss in <https://ieeexplore.ieee.org/document/8237429>)

        NOTE: Again, the loss is ambiguous to be squared distance or euclidian distance.
        Also, it is not explicitly mentioned that the loss is scaled by B,W.
        We assumed the loss is squared distance, and scaled by B,W (by taking mean instead of sum) as it produced closed loss values reported in the paper.


        inputs:
            VGG_features_style: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # define style loss for each term
        style_loss_each_term = lambda A1, A2 : torch.mean(torch.square(torch.sub(A1.mean([2,3]), A2.mean([2,3])))) + \
                                               torch.mean(torch.square(torch.sub(A1.std([2,3]), A2.std([2,3]))))


        # TODO: Check if the instance normalizing scale is correct
        # calculate style loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1
        style_loss =    style_loss_each_term(VGG_features_style[0], VGG_features_output[0]) + \
                        style_loss_each_term(VGG_features_style[1], VGG_features_output[1]) + \
                        style_loss_each_term(VGG_features_style[2], VGG_features_output[2]) + \
                        style_loss_each_term(VGG_features_style[3], VGG_features_output[3])
        return style_loss



    

    # TODO: Implement the other loss in the paper (similarity loss)
    def get_similarity_loss(self, VGG_features_content, VGG_features_output):
        """
        calculates the similarity loss defined in the paper

        inputs:
            VGG_features_content: list of 2 tensors, each tensor is the output of relu 4_1, relu 5_1 layers from the content image
            VGG_features_output: list of 2 tensors, each tensor is the output of relu 4_1, relu 5_1 layers from the output image
        """
        
        # get the scaled self cosine distance map for the content and output images
        scaled_self_cosine_distance_map_content_relu_4_1 = get_scaled_self_cosine_distance_map(VGG_features_content[0])
        scaled_self_cosine_distance_map_output_relu_4_1 = get_scaled_self_cosine_distance_map(VGG_features_output[0])

        scaled_self_cosine_distance_map_content_relu_5_1 = get_scaled_self_cosine_distance_map(VGG_features_content[1])
        scaled_self_cosine_distance_map_output_relu_5_1 = get_scaled_self_cosine_distance_map(VGG_features_output[1])


        # calculate the similarity loss, dividing with {n_{x}}^{2}, as well as batch sizes
        similarity_loss = torch.sum(torch.abs(torch.sub(scaled_self_cosine_distance_map_content_relu_4_1, scaled_self_cosine_distance_map_output_relu_4_1))) / (scaled_self_cosine_distance_map_content_relu_4_1.shape[0] * scaled_self_cosine_distance_map_content_relu_4_1.shape[1]) + \
                          torch.sum(torch.abs(torch.sub(scaled_self_cosine_distance_map_content_relu_5_1, scaled_self_cosine_distance_map_output_relu_5_1))) / (scaled_self_cosine_distance_map_content_relu_5_1.shape[0] * scaled_self_cosine_distance_map_content_relu_5_1.shape[1])


        return similarity_loss





if __name__ == "__main__":

    import cv2
    import numpy as np
    import matplotlib.pyplot as plt


    # define a function to preprocess the image
    def preprocess_image(image):
        image = cv2.resize(image, (256, 256))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    IF_OUTPUT_CONTENT_AND_STYLE_LOSS = True
    IF_OUTPUT_SIMILARITY_LOSS = True



    # get the absolute path of the project
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    
    # create an instance of the custom loss class
    custom_loss_instance = custom_loss(project_absolute_path = project_absolute_path,
                                       feature_extractor_model_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                                       default_lambda_value=10)
    



    # test with figure 9 images from the paper
    content_image_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/content_layer.png"))
    style_image_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/style_layer.png"))
    output_image_1_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_1.png"))
    output_image_3_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_3.png"))
    output_image_5_figure_9_raw_image = cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/figure9/output_layer_5.png"))

    content_image_figure_9 = preprocess_image(content_image_figure_9_raw_image)
    style_image_figure_9 = preprocess_image(style_image_figure_9_raw_image)
    output_image_1_figure_9 = preprocess_image(output_image_1_figure_9_raw_image)
    output_image_3_figure_9 = preprocess_image(output_image_3_figure_9_raw_image)
    output_image_5_figure_9 = preprocess_image(output_image_5_figure_9_raw_image)

    # calculate total loss for output_image_1
    total_loss_1_figure_9, content_loss_1_figure_9, style_loss_1_figure_9, similarity_loss_1_figure_9 = custom_loss_instance(content_image_figure_9,
                                                                                                 style_image_figure_9,
                                                                                                 output_image_1_figure_9,
                                                                                                 output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                                                 output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)


    # calculate total loss for output_image_3
    total_loss_3_figure_9, content_loss_3_figure_9, style_loss_3_figure_9, similarity_loss_3_figure_9 = custom_loss_instance(content_image_figure_9,
                                                                                                 style_image_figure_9,
                                                                                                 output_image_3_figure_9,
                                                                                                 output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                                                 output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)


    # calculate total loss for output_image_5
    total_loss_5_figure_9, content_loss_5_figure_9, style_loss_5_figure_9, similarity_loss_5_figure_9 = custom_loss_instance(content_image_figure_9,
                                                                                                 style_image_figure_9,
                                                                                                 output_image_5_figure_9,
                                                                                                 output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                                                                 output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)





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
    ax[0, 2].text(300, 50, f"From figure 9, layer 1 output of the paper", fontsize=25, color="green")
    ax[0, 2].text(300, 100, f"Total Loss: {total_loss_1_figure_9.item():.5}", fontsize=25, color="red")
    ax[0, 2].text(300, 150, f"Content Loss: {content_loss_1_figure_9.item():.5}", fontsize=25, color="red")
    ax[0, 2].text(300, 200, f"Style Loss: {style_loss_1_figure_9.item():.5}", fontsize=25, color="red")
    ax[0, 2].text(300, 250, f"Similarity Loss: {similarity_loss_1_figure_9.item():.5}", fontsize=25, color="red")



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
    ax[1, 2].text(300, 50, f"From figure 9, layer 3 output of the paper", fontsize=25, color="green")
    ax[1, 2].text(300, 100, f"Total Loss: {total_loss_3_figure_9.item():.5}", fontsize=25, color="red")
    ax[1, 2].text(300, 150, f"Content Loss: {content_loss_3_figure_9.item():.5}", fontsize=25, color="red")
    ax[1, 2].text(300, 200, f"Style Loss: {style_loss_3_figure_9.item():.5}", fontsize=25, color="red")
    ax[1, 2].text(300, 250, f"Similarity Loss: {similarity_loss_3_figure_9.item():.5}", fontsize=25, color="red")



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
    ax[2, 2].text(300, 50, f"From figure 9, layer 5 output of the paper", fontsize=25, color="green")
    ax[2, 2].text(300, 100, f"Total Loss: {total_loss_5_figure_9.item():.5}", fontsize=25, color="red")
    ax[2, 2].text(300, 150, f"Content Loss: {content_loss_5_figure_9.item():.5}", fontsize=25, color="red")
    ax[2, 2].text(300, 200, f"Style Loss: {style_loss_5_figure_9.item():.5}", fontsize=25, color="red")
    ax[2, 2].text(300, 250, f"Similarity Loss: {similarity_loss_5_figure_9.item():.5}", fontsize=25, color="red")

    plt.tight_layout()


    # save the figure
    fig.savefig("codes/images_to_try_loss_function/figure9_losses.png")

    
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
        content_image = preprocess_image(content_image_raw_image)
        style_image = preprocess_image(style_image_raw_image)
        output_AdaAttN_image = preprocess_image(output_AdaAttN_image_raw_image)
        output_Master_ZS_layer1_image = preprocess_image(output_Master_ZS_layer1_image_raw_image)
        output_Master_ZS_layer3_image = preprocess_image(output_Master_ZS_layer3_image_raw_image)
        output_Master_FS_image = preprocess_image(output_Master_FS_image_raw_image)

        # calculate total loss for output_AdaAttN_image
        losses_AdaAttN          = custom_loss_instance(content_image,
                                                       style_image,
                                                       output_AdaAttN_image,
                                                       output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                       output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
        losses_Master_ZS_layer1 = custom_loss_instance(content_image,
                                                       style_image,
                                                       output_Master_ZS_layer1_image,
                                                       output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                       output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
        losses_Master_ZS_layer3 = custom_loss_instance(content_image,
                                                       style_image,
                                                       output_Master_ZS_layer3_image,
                                                       output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                       output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
        losses_Master_FS        = custom_loss_instance(content_image,
                                                       style_image,
                                                       output_Master_FS_image,
                                                       output_content_and_style_loss=IF_OUTPUT_CONTENT_AND_STYLE_LOSS,
                                                       output_similarity_loss=IF_OUTPUT_SIMILARITY_LOSS)
        
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
        ax[1, 0].text(300, 100, f"Total Loss: {losses_AdaAttN[0].item():.5}", fontsize=25, color="red")
        if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
            ax[1, 0].text(300, 150, f"Content Loss: {losses_AdaAttN[1].item():.5}", fontsize=25, color="red")
            ax[1, 0].text(300, 200, f"Style Loss: {losses_AdaAttN[2].item():.5}", fontsize=25, color="red")
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[1, 0].text(300, 250, f"Similarity Loss: {losses_AdaAttN[3].item():.5}", fontsize=25, color="red")
        else:
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[1, 0].text(300, 150, f"Similarity Loss: {losses_AdaAttN[1].item():.5}", fontsize=25, color="red")
        
        # make ax[1, 1] empty
        ax[1, 1].axis("off")


        # show the Master_ZS_layer1 output image
        ax[2, 0].imshow(cv2.cvtColor(output_Master_ZS_layer1_image_raw_image, cv2.COLOR_BGR2RGB))
        ax[2, 0].set_title("Master ZS Layer 1 Output")
        ax[2, 0].axis("off")
        ax[2, 0].text(300, 50, f"Master ZS Layer 1 Output", fontsize=25, color="green")
        ax[2, 0].text(300, 100, f"Total Loss: {losses_Master_ZS_layer1[0].item():.5}", fontsize=25, color="red")
        if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
            ax[2, 0].text(300, 150, f"Content Loss: {losses_Master_ZS_layer1[1].item():.5}", fontsize=25, color="red")
            ax[2, 0].text(300, 200, f"Style Loss: {losses_Master_ZS_layer1[2].item():.5}", fontsize=25, color="red")
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[2, 0].text(300, 250, f"Similarity Loss: {losses_Master_ZS_layer1[3].item():.5}", fontsize=25, color="red")
        else:
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[2, 0].text(300, 150, f"Similarity Loss: {losses_Master_ZS_layer1[1].item():.5}", fontsize=25, color="red")
        
        # make ax[2, 1] empty
        ax[2, 1].axis("off")


        # show the Master_ZS_layer3 output image
        ax[3, 0].imshow(cv2.cvtColor(output_Master_ZS_layer3_image_raw_image, cv2.COLOR_BGR2RGB))
        ax[3, 0].set_title("Master ZS Layer 3 Output")
        ax[3, 0].axis("off")
        ax[3, 0].text(300, 50, f"Master ZS Layer 3 Output", fontsize=25, color="green")
        ax[3, 0].text(300, 100, f"Total Loss: {losses_Master_ZS_layer3[0].item():.5}", fontsize=25, color="red")
        if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
            ax[3, 0].text(300, 150, f"Content Loss: {losses_Master_ZS_layer3[1].item():.5}", fontsize=25, color="red")
            ax[3, 0].text(300, 200, f"Style Loss: {losses_Master_ZS_layer3[2].item():.5}", fontsize=25, color="red")
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[3, 0].text(300, 250, f"Similarity Loss: {losses_Master_ZS_layer3[3].item():.5}", fontsize=25, color="red")
        else:
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[3, 0].text(300, 150, f"Similarity Loss: {losses_Master_ZS_layer3[1].item():.5}", fontsize=25, color="red")
        
        # make ax[3, 1] empty
        ax[3, 1].axis("off")


        # show the Master_FS output image
        ax[4, 0].imshow(cv2.cvtColor(output_Master_FS_image_raw_image, cv2.COLOR_BGR2RGB))
        ax[4, 0].set_title("Master FS Output")
        ax[4, 0].axis("off")
        ax[4, 0].text(300, 50, f"Master FS Output", fontsize=25, color="green")
        ax[4, 0].text(300, 100, f"Total Loss: {losses_Master_FS[0].item():.5}", fontsize=25, color="red")
        if IF_OUTPUT_CONTENT_AND_STYLE_LOSS:
            ax[4, 0].text(300, 150, f"Content Loss: {losses_Master_FS[1].item():.5}", fontsize=25, color="red")
            ax[4, 0].text(300, 200, f"Style Loss: {losses_Master_FS[2].item():.5}", fontsize=25, color="red")
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[4, 0].text(300, 250, f"Similarity Loss: {losses_Master_FS[3].item():.5}", fontsize=25, color="red")
        else:
            if IF_OUTPUT_SIMILARITY_LOSS:
                ax[4, 0].text(300, 150, f"Similarity Loss: {losses_Master_FS[1].item():.5}", fontsize=25, color="red")
        
        # make ax[4, 1] empty
        ax[4, 1].axis("off")




        plt.tight_layout()

        # save the figure
        fig.savefig(f"codes/images_to_try_loss_function/figure4_losses_column{column_index}.png")

        # close the figure
        plt.close(fig)

