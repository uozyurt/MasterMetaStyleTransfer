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

class Test:
    def __init__(
        self,
        content_images_path: str,
        style_images_path: str,
        if_calculate_similatiry_loss: bool = False,
        output_path: str = "",
        test_transform: transforms.Compose = None,
        use_imagenet_normalization_for_swin: bool = True,
        use_imagenet_normalization_for_loss: bool = False,
    ):
        self.content_images_path = content_images_path
        self.style_images_path = style_images_path
        self.if_calculate_similatiry_loss = if_calculate_similatiry_loss
        self.use_imagenet_normalization_for_swin = use_imagenet_normalization_for_swin
        self.use_imagenet_normalization_for_loss = use_imagenet_normalization_for_loss
        self.output_path = output_path

        if self.output_path != "":
            if not os.path.exists(output_path):
                os.makedirs(output_path)

        if test_transform is None:
            self.test_transform = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((256, 256)), # -> resize to 512x512
                transforms.ToTensor()
            ])
        else:
            self.test_transform = test_transform

        self.imagenet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


    def test_(self,
             master_style_transfer_model,
             loss_instance,
             test_device: str = "cuda",
             transformer_layer_count: int = 1,
             ):


        # set the loss instance to evaluation mode
        loss_instance.eval()

        # set the model to evaluation mode
        master_style_transfer_model.eval()


        # load the loss instance to the device
        loss_instance = loss_instance.to(test_device)

        # load the model to the device
        master_style_transfer_model = master_style_transfer_model.to(test_device)


        # get the content image list
        content_image_list = glob.glob(os.path.join(self.content_images_path, "*"))

        # get the style image list
        style_image_list = glob.glob(os.path.join(self.style_images_path, "*"))


        # open all images and preprocess them (if test set is too big, ram may not be enough to hold all images at once for caching)
        processed_content_images = []
        processed_style_images = []

        for image_path in content_image_list:
            processed_content_images.append(self.test_transform(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)))

        for image_path in style_image_list:
            processed_style_images.append(self.test_transform(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)))


        # create a loss lists
        total_loss_list = []
        content_loss_list = []
        style_loss_list = []

        if self.if_calculate_similatiry_loss:
            similarity_loss_list = []


        # iterate over all combinations of content and style images
        for content_image, content_image_path in zip(processed_content_images, content_image_list):

            # get the content image name wihtout the extension
            content_image_name = os.path.basename(content_image_path).split(".")[0]

            # load the content image
            content_image = content_image.unsqueeze(0)

            # if imagenet normalization is used for swin, apply it
            if self.use_imagenet_normalization_for_swin:
                content_image_normalized = self.imagenet_normalization(content_image).to(test_device)
            else:
                content_image = content_image.to(test_device)





            for style_image, style_image_path in zip(processed_style_images, style_image_list):

                # load the style image
                style_image = style_image.unsqueeze(0)

                if self.use_imagenet_normalization_for_swin:
                    style_image_normalized = self.imagenet_normalization(style_image).to(test_device)
                else:
                    style_image = style_image.to(test_device)

                # get the style image name wihtout the extension
                style_image_name = os.path.basename(style_image_path).split(".")[0]

                # get the output image name
                output_image_name = f"{content_image_name}_stylized_with_{style_image_name}_layers_{transformer_layer_count}.jpg"

                # get the output image path
                output_image_path = os.path.join(self.output_path, output_image_name)

                # get the stylized image
                with torch.no_grad():
                    if self.use_imagenet_normalization_for_swin:
                        stylized_image = master_style_transfer_model(content_image_normalized,
                                                                     style_image_normalized,
                                                                     transformer_layer_count)
                    else:
                        stylized_image = master_style_transfer_model(content_image,
                                                                     style_image,
                                                                     transformer_layer_count)


                    if self.use_imagenet_normalization_for_loss:
                        if self.use_imagenet_normalization_for_swin:
                            # calculate the loss (normalize output image with imagenet normalization)
                            loss = loss_instance(content_image_normalized,
                                                 style_image_normalized,
                                                 self.imagenet_normalization(stylized_image),
                                                 output_content_and_style_loss=True,
                                                 output_similarity_loss=self.if_calculate_similatiry_loss)
                        else:
                            # calculate the loss (normalize all images with imagenet normalization)
                            loss = loss_instance(self.imagenet_normalization(content_image),
                                                 self.imagenet_normalization(style_image),
                                                 self.imagenet_normalization(stylized_image),
                                                 output_content_and_style_loss=True,
                                                 output_similarity_loss=self.if_calculate_similatiry_loss)
                    else:
                        if self.use_imagenet_normalization_for_swin:
                            content_image_normalized = content_image_normalized.to("cpu")
                            style_image_normalized = style_image_normalized.to("cpu")

                            content_image = content_image.to(self.device)
                            style_image = style_image.to(self.device)

                            # calculate the loss (normalize output image with imagenet normalization)
                            loss = loss_instance(content_image,
                                                 style_image,
                                                 stylized_image,
                                                 output_content_and_style_loss=True,
                                                 output_similarity_loss=self.if_calculate_similatiry_loss)
                            
                            content_image = content_image.to("cpu")
                            style_image = style_image.to("cpu")
                        else:
                            # calculate the loss (normalize all images with imagenet normalization)
                            loss = loss_instance(content_image,
                                                 style_image,
                                                 stylized_image,
                                                 output_content_and_style_loss=True,
                                                 output_similarity_loss=self.if_calculate_similatiry_loss)

                    if self.if_calculate_similatiry_loss:
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

                
                if self.output_path != "":
                    # save the stylized image
                    cv2.imwrite(output_image_path, np.clip(stylized_image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255, 0, 255).astype(np.uint8))



        if self.if_calculate_similatiry_loss:
            return total_loss_list, content_loss_list, style_loss_list, similarity_loss_list
        else:
            return total_loss_list, content_loss_list, style_loss_list

        




        




