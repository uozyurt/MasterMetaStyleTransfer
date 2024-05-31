import torch
import cv2
import numpy as np
from torchvision import transforms

import os
import glob
import sys
project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

sys.path.append(project_absolute_path)


from codes.full_model import MasterStyleTransferModel
from codes.loss import custom_loss


# define the flags
CALCULATE_STYLE_CONTENT_LOSS = True
CALCULATE_SIMILARITY_LOSS = False

SAVE_OUTPUTS = True



# test folder paths
content_images_path = "test/content_input"
style_images_path = "test/style_input"



# pretrained model paths
style_transformer_path = "test/pretrained_weights/experiment_46_decoder_11000.pt"
decoder_path = "test/pretrained_weights/experiment_46_style_transformer_11000.pt"

output_path = "test/output_images"

if not os.path.exists(output_path):
    os.makedirs(output_path)


test_transform = transforms.Compose([
    transforms.ToPILImage(), # -> PIL image
    transforms.Resize((256, 256)), # -> resize to 512x512
    transforms.ToTensor()
])

imagenet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

test_device = "cuda" if torch.cuda.is_available() else "cpu"

if CALCULATE_STYLE_CONTENT_LOSS:
    # create the loss instance
    loss_instance = custom_loss(
        project_absolute_path=project_absolute_path,
        feature_extractor_model_relative_path="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
        use_vgg19_with_batchnorm=False,
        default_lambda_value=10.0,
        distance_content="euclidian_squared",
        distance_style="euclidian"
    )

    # set the loss instance to evaluation mode
    loss_instance.eval()

# load the loss instance to the device
loss_instance = loss_instance.to(test_device)


# load the model
master_style_transfer_model = MasterStyleTransferModel(
    project_absolute_path = project_absolute_path,
    swin_model_relative_path = "weights/swin_B_first_2_stages.pt",
    swin_variant = "swin_B",
    style_encoder_dim = 256,
    style_decoder_dim = 256,
    style_encoder_num_heads = 8,
    style_decoder_num_heads = 8,
    style_encoder_window_size = [7, 7],
    style_decoder_window_size = [7, 7],
    style_encoder_shift_size = [4, 4],
    style_decoder_shift_size = [4, 4],
    style_encoder_mlp_ratio = 4.0,
    style_decoder_mlp_ratio = 4.0,
    style_encoder_dropout = 0.0,
    style_decoder_dropout = 0.0,
    style_encoder_attention_dropout = 0.0,
    style_decoder_attention_dropout = 0.0,
    style_encoder_qkv_bias = True,
    style_decoder_qkv_bias = True,
    style_encoder_proj_bias = True,
    style_decoder_proj_bias = True,
    style_encoder_stochastic_depth_prob = 0.1,
    style_decoder_stochastic_depth_prob = 0.1,
    style_encoder_norm_layer = None,
    style_decoder_norm_layer = torch.nn.LayerNorm,
    style_encoder_MLP_activation_layer = torch.nn.GELU,
    style_decoder_MLP_activation_layer = torch.nn.GELU,
    style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation = True,
    style_decoder_use_instance_norm_with_affine = False,
    style_decoder_use_regular_MHA_instead_of_Swin_at_the_end = False,
    style_decoder_use_Key_instance_norm_after_linear_transformation = True,
    style_decoder_exclude_MLP_after_Fcs_self_MHA = False,
    style_transformer_load_pretrained_weights = False,
    direct_pretrained_style_transformer_path = style_transformer_path,
    direct_pretrained_decoder_path = decoder_path,
)

# set the model to evaluation mode
master_style_transfer_model.eval()

# load the model to the device
master_style_transfer_model = master_style_transfer_model.to(test_device)


# get the content image list
content_image_list = glob.glob(os.path.join(content_images_path, "*"))

# get the style image list
style_image_list = glob.glob(os.path.join(style_images_path, "*"))

trial = test_transform(cv2.cvtColor(cv2.imread(content_image_list[0]), cv2.COLOR_BGR2RGB))

# open all images and preprocess them

processed_content_images = []
processed_style_images = []

for image_path in content_image_list:
    processed_content_images.append(test_transform(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)))

for image_path in style_image_list:
    processed_style_images.append(test_transform(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)))


if CALCULATE_STYLE_CONTENT_LOSS:
    # create a loss list
    total_loss_list = []
    content_loss_list = []
    style_loss_list = []

    if CALCULATE_SIMILARITY_LOSS:
        similarity_loss_list = []


# DETERMINE THE TRANSFORMER LAYER COUNT TO BE USED
transformer_layer_count = 1

# iterate over all combinations of content and style images
for content_image, content_image_path in zip(processed_content_images, content_image_list):

    # load the content image
    content_image = content_image.unsqueeze(0).to(test_device)

    # get the content image name wihtout the extension
    content_image_name = os.path.basename(content_image_path).split(".")[0]


    for style_image, style_image_path in zip(processed_style_images, style_image_list):

        # load the style image
        style_image = style_image.unsqueeze(0).to(test_device)

        # get the style image name wihtout the extension
        style_image_name = os.path.basename(style_image_path).split(".")[0]

        # get the output image name
        output_image_name = f"{content_image_name}_stylized_with_{style_image_name}_layers_{transformer_layer_count}.jpg"

        # get the output image path
        output_image_path = os.path.join(output_path, output_image_name)

        # get the stylized image
        with torch.no_grad():
            stylized_image = master_style_transfer_model(content_image, style_image)
        
            if CALCULATE_STYLE_CONTENT_LOSS:
                # calculate the loss
                loss = loss_instance(content_image,
                                     stylized_image,
                                     style_image,
                                     output_content_and_style_loss=True,
                                     output_similarity_loss=CALCULATE_SIMILARITY_LOSS)

                if CALCULATE_SIMILARITY_LOSS:
                    total_loss, content_loss, style_loss, similarity_loss = loss

                    total_loss_list.append(total_loss.item())
                    content_loss_list.append(content_loss.item())
                    style_loss_list.append(style_loss.item())
                    similarity_loss_list.append(similarity_loss.item())
                else:
                    total_loss, content_loss, style_loss = loss

                    total_loss_list.append(total_loss.item())
                    content_loss_list.append(content_loss.item())
                    style_loss_list.append(style_loss.item())

        

        if SAVE_OUTPUTS:
            # save the stylized image
            cv2.imwrite(output_image_path, np.clip(stylized_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8))


if CALCULATE_STYLE_CONTENT_LOSS:
    total_loss_mean = np.mean(total_loss_list)
    total_loss_std = np.std(total_loss_list)
    total_loss_confidence_interval = 1.96 * total_loss_std / np.sqrt(len(total_loss_list))

    content_loss_mean = np.mean(content_loss_list)
    content_loss_std = np.std(content_loss_list)
    content_loss_confidence_interval = 1.96 * content_loss_std / np.sqrt(len(content_loss_list))

    style_loss_mean = np.mean(style_loss_list)
    style_loss_std = np.std(style_loss_list)
    style_loss_confidence_interval = 1.96 * style_loss_std / np.sqrt(len(style_loss_list))

    print(f"Total Loss Mean: {total_loss_mean:.2f}, Total Loss Std: {total_loss_std:.2f}")
    print(f"Total loss 95% CI: [{total_loss_mean - total_loss_confidence_interval:.2f}, {total_loss_mean + total_loss_confidence_interval:.2f}]")
    print("\n")
    print(f"Content Loss Mean: {content_loss_mean:.2f}, Content Loss Std: {content_loss_std:.2f}")
    print(f"Content loss 95% CI: [{content_loss_mean - content_loss_confidence_interval:.2f}, {content_loss_mean + content_loss_confidence_interval:.2f}]")
    print("\n")
    print(f"Style Loss Mean: {style_loss_mean:.2f}, Style Loss Std: {style_loss_std:.2f}")
    print(f"Style loss 95% CI: [{style_loss_mean - style_loss_confidence_interval:.2f}, {style_loss_mean + style_loss_confidence_interval:.2f}]")
    print("\n")

    if CALCULATE_SIMILARITY_LOSS:
        similarity_loss_mean = np.mean(similarity_loss_list)
        similarity_loss_std = np.std(similarity_loss_list)
        similarity_loss_confidence_interval = 1.96 * similarity_loss_std / np.sqrt(len(similarity_loss_list))

        print(f"Similarity Loss Mean: {similarity_loss_mean:.2f}, Similarity Loss Std: {similarity_loss_std:.2f}")
        print(f"Similarity loss 95% CI: [{similarity_loss_mean - similarity_loss_confidence_interval:.2f}, {similarity_loss_mean + similarity_loss_confidence_interval:.2f}]")
        print("\n")

        




        




