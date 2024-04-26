import torch
import os
import torch.nn as nn



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
                 feature_extractor_model_relative_path="models/vgg_19_last_layer_is_relu_5_1_output.pt",
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
            from codes.utils.download_VGG19_create_cutted_model_to_process import download_VGG19_and_create_cutted_model_to_process

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
    def forward(self, content_image, style_image, output_image):
        """
        Gets the content image, style image, and output image, and returns the total loss (content loss + lambda * style loss)
        All images should be in the exact same shape: [batch_size, 3, 256, 256]
        """
        return self.get_overall_loss(content_image = content_image,
                                     style_image = style_image,
                                     output_image = output_image,
                                     loss_weight = self.lambda_value)

    # Content Loss
    def get_content_loss(self, VGG_features_content, VGG_features_output):
        """
        calculates the content loss (normalized perceptual loss in <https://arxiv.org/pdf/1603.08155>)

        inputs:
        VGG_features_content: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
        VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # define content loss for each term
        content_loss_each_term = lambda A1, A2, instance_norm: torch.dist(instance_norm(A1), instance_norm(A2), p=2)

        # get the shapes of the tensors
        features_shape = VGG_features_content[0].shape

        # TODO: Check if the instance normalizing scale is correct
        # calculate content loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1 (also scaled by W,H,C, as in the mentioned paper)
        content_loss =  content_loss_each_term(VGG_features_content[0], VGG_features_output[0], nn.InstanceNorm2d(128)) / (features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3]) + \
                        content_loss_each_term(VGG_features_content[1], VGG_features_output[1], nn.InstanceNorm2d(256)) / (features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3]) + \
                        content_loss_each_term(VGG_features_content[2], VGG_features_output[2], nn.InstanceNorm2d(512)) / (features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3]) + \
                        content_loss_each_term(VGG_features_content[3], VGG_features_output[3], nn.InstanceNorm2d(512)) / (features_shape[0] * features_shape[1] * features_shape[2] * features_shape[3])
        

        return content_loss

    # Style Loss
    def get_style_loss(self, VGG_features_style, VGG_features_output):
        """
        calculates the style loss (mean-variance loss in <https://ieeexplore.ieee.org/document/8237429>)

        inputs:
        VGG_features_style: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the content image
        VGG_features_output: list of 4 tensors, each tensor is the output of relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers from the output image
        """

        # define style loss for each term
        style_loss_each_term = lambda A1, A2 : torch.dist(A1.mean([2,3]), A2.mean([2,3]), p=2) + torch.dist(A1.std([2,3]), A2.std([2,3]), p=2)

        # get the shapes of the tensors
        features_shape = VGG_features_style[0].shape

        # TODO: Check if the instance normalizing scale is correct
        # calculate style loss for relu 2_1, relu 3_1, relu 4_1, relu 5_1
        style_loss =    style_loss_each_term(VGG_features_style[0], VGG_features_output[0]) / (features_shape[0] * features_shape[1]) + \
                        style_loss_each_term(VGG_features_style[1], VGG_features_output[1]) / (features_shape[0] * features_shape[1]) + \
                        style_loss_each_term(VGG_features_style[2], VGG_features_output[2]) / (features_shape[0] * features_shape[1]) + \
                        style_loss_each_term(VGG_features_style[3], VGG_features_output[3]) / (features_shape[0] * features_shape[1])    
        return style_loss


    # Overall, weighted loss (containin both content and style loss)
    def get_overall_loss(self, content_image, style_image, output_image, loss_weight=None):
        """
        This function calculates the total loss (content loss + lambda * style loss) for the output image.
        It uses the custom VGG19 model to get the outputs from relu 2_1, relu 3_1, relu 4_1, relu 5_1 layers, as it is declared in the paper.
        """
        # inputs are in shape: [batch_size, 3, 256, 256]

        # check if lambda value is given
        if loss_weight is None:
            loss_weight = self.lambda_value

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

        print(f"Content loss: {content_loss.item():.10f}, Style loss: {style_loss.item():.10f}")
        
        # calculate total loss
        total_loss = content_loss + loss_weight * style_loss

        return total_loss
        
        
    # TODO: Implement the other loss in the paper (similarity loss)
    def get_similarity_loss(self, VGG_features_content, VGG_features_style, VGG_features_output):
        pass










if __name__ == "__main__":

    import cv2
    import numpy as np
    
    # define a function to preprocess the image
    def preprocess_image(image):
        image = cv2.resize(image, (256, 256))
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255
        image = torch.tensor(image, dtype=torch.float32)
        return image
    
    # get the absolute path of the project
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    
    # create an instance of the custom loss class
    custom_loss_instance = custom_loss(project_absolute_path = project_absolute_path,
                                       feature_extractor_model_relative_path="models/vgg_19_last_layer_is_relu_5_1_output.pt",
                                       default_lambda_value=10)
    

    # preprocess the example images
    content_image = preprocess_image(cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/content_layer.png")))
    style_image = preprocess_image(cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/style_layer.png")))
    output_image_1 = preprocess_image(cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/output_layer_1.png")))
    output_image_3 = preprocess_image(cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/output_layer_3.png")))
    output_image_5 = preprocess_image(cv2.imread(os.path.join(project_absolute_path, "codes/images_to_try_loss_function/output_layer_5.png")))

    # calculate total loss for output_image_1
    total_loss_1 = custom_loss_instance(content_image, style_image, output_image_1)
    print(f"Total loss for output_image_1: {total_loss_1.item():.10f}")
    print()

    # calculate total loss for output_image_3
    total_loss_3 = custom_loss_instance(content_image, style_image, output_image_3)
    print(f"Total loss for output_image_3: {total_loss_3.item():.10f}")
    print()

    # calculate total loss for output_image_5
    total_loss_5 = custom_loss_instance(content_image, style_image, output_image_5)
    print(f"Total loss for output_image_5: {total_loss_5.item():.10f}")
    print()

