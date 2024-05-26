import argparse
import os
import yaml
import random
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
import wandb

from codes.full_model import MasterStyleTransferModel
from codes.loss import custom_loss
from codes.get_dataloader import coco_train_dataset, wikiart_dataset, InfiniteSampler



class Train:
    def __init__(self, config):
        if config.set_seed:
            np.random.seed(config.seed)
            random.seed(config.seed)
            torch.manual_seed(config.seed)
            torch.cuda.manual_seed_all(config.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(config.seed)

            print(f'Using seed {config.seed}')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # project path 
        self.project_root = config.project_root
        self.model_save_path = config.model_save_path


        # Dataset paths
        self.coco_dataset_path = config.coco_dataset_path
        self.wikiart_dataset_path = config.wikiart_dataset_path


        # Loss model path
        self.loss_model_path = config.loss_model_path
        self.use_vgg19_with_batchnorm = config.use_vgg19_with_batchnorm


        # Dataloader parameters
        self.batch_size_style = config.batch_size_style
        self.batch_size_content = config.batch_size_content
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle
        self.use_infinite_sampler = config.use_infinite_sampler
        self.pin_memory = config.pin_memory


        # Hyperparameters
        self.freeze_encoder = config.freeze_encoder
        self.inner_lr = config.inner_lr
        self.outer_lr = config.outer_lr
        self.num_inner_updates = config.num_inner_updates
        self.max_layers = config.max_layers
        self.lambda_style = config.lambda_style
        self.loss_distance = config.loss_distance
        self.use_imagenet_normalization_for_swin = config.use_imagenet_normalization_for_swin
        self.use_imagenet_normalization_for_loss = config.use_imagenet_normalization_for_loss
        self.save_every = config.save_every
        self.max_iterations = config.max_iterations



        # MasterStyleTransferModel parameters
        self.swin_model_relative_path = config.swin_model_relative_path
        self.swin_variant = config.swin_variant
        self.style_encoder_dim = config.style_encoder_dim
        self.style_decoder_dim = config.style_decoder_dim
        self.style_encoder_num_heads = config.style_encoder_num_heads
        self.style_decoder_num_heads = config.style_decoder_num_heads
        self.style_encoder_window_size = config.style_encoder_window_size
        self.style_decoder_window_size = config.style_decoder_window_size
        self.style_encoder_shift_size = config.style_encoder_shift_size
        self.style_decoder_shift_size = config.style_decoder_shift_size
        self.style_encoder_mlp_ratio = config.style_encoder_mlp_ratio
        self.style_decoder_mlp_ratio = config.style_decoder_mlp_ratio
        self.style_encoder_dropout = config.style_encoder_dropout
        self.style_decoder_dropout = config.style_decoder_dropout
        self.style_encoder_attention_dropout = config.style_encoder_attention_dropout
        self.style_decoder_attention_dropout = config.style_decoder_attention_dropout
        self.style_encoder_qkv_bias = config.style_encoder_qkv_bias
        self.style_decoder_qkv_bias = config.style_decoder_qkv_bias
        self.style_encoder_proj_bias = config.style_encoder_proj_bias
        self.style_decoder_proj_bias = config.style_decoder_proj_bias
        self.style_encoder_stochastic_depth_prob = config.style_encoder_stochastic_depth_prob
        self.style_decoder_stochastic_depth_prob = config.style_decoder_stochastic_depth_prob
        self.style_encoder_norm_layer = config.style_encoder_norm_layer
        self.style_decoder_norm_layer = config.style_decoder_norm_layer
        self.style_encoder_MLP_activation_layer = config.style_encoder_MLP_activation_layer
        self.style_decoder_MLP_activation_layer = config.style_decoder_MLP_activation_layer
        self.style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation = config.style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation
        self.style_decoder_use_instance_norm_with_affine = config.style_decoder_use_instance_norm_with_affine
        self.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end = config.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end
        self.style_decoder_use_Key_instance_norm_after_linear_transformation = config.style_decoder_use_Key_instance_norm_after_linear_transformation
        self.style_decoder_exclude_MLP_after_Fcs_self_MHA = config.style_decoder_exclude_MLP_after_Fcs_self_MHA
        self.decoder_initializer = config.decoder_initializer



        # Seed configuration
        self.set_seed = config.set_seed
        self.seed = config.seed



        # Verbose
        self.verbose = config.verbose



        # Wandb parameters
        self.use_wandb = config.use_wandb
        self.online = config.online
        self.exp_name = config.exp_name


        # Make sure model saving path exists
        if not os.path.exists(os.path.join(self.model_save_path, self.exp_name)):
            os.makedirs(os.path.join(self.model_save_path, self.exp_name))
        else:
            # If the model saving path already exists, create a new folder with a new name and change experiment name
            print(f"Model saving path already exists: {os.path.join(self.model_save_path, self.exp_name)}")

            self.exp_name = self.exp_name + "_new_0"
            while os.path.exists(os.path.join(self.model_save_path, self.exp_name)):
                self.exp_name = self.exp_name[:-1] + str(int(self.exp_name[-1]) + 1)
            
            print(f"New experiment name: {self.exp_name}")

            os.makedirs(os.path.join(self.model_save_path, self.exp_name))
            
        
        # save config file as a yaml file
        with open(os.path.join(self.project_root, self.model_save_path, self.exp_name, f"{self.exp_name}_config.yaml"), 'w') as file:
            yaml.dump(vars(self), file)


        # Initialize the master style transfer model
        self.master_style_transformer = MasterStyleTransferModel(
            project_absolute_path=self.project_root,
            swin_model_relative_path=self.swin_model_relative_path,
            swin_variant=self.swin_variant,
            style_encoder_dim=self.style_encoder_dim,
            style_decoder_dim=self.style_decoder_dim,
            style_encoder_num_heads=self.style_encoder_num_heads,
            style_decoder_num_heads=self.style_decoder_num_heads,
            style_encoder_window_size=self.style_encoder_window_size,
            style_decoder_window_size=self.style_decoder_window_size,
            style_encoder_shift_size=self.style_encoder_shift_size,
            style_decoder_shift_size=self.style_decoder_shift_size,
            style_encoder_mlp_ratio=self.style_encoder_mlp_ratio,
            style_decoder_mlp_ratio=self.style_decoder_mlp_ratio,
            style_encoder_dropout=self.style_encoder_dropout,
            style_decoder_dropout=self.style_decoder_dropout,
            style_encoder_attention_dropout=self.style_encoder_attention_dropout,
            style_decoder_attention_dropout=self.style_decoder_attention_dropout,
            style_encoder_qkv_bias=self.style_encoder_qkv_bias,
            style_decoder_qkv_bias=self.style_decoder_qkv_bias,
            style_encoder_proj_bias=self.style_encoder_proj_bias,
            style_decoder_proj_bias=self.style_decoder_proj_bias,
            style_encoder_stochastic_depth_prob=self.style_encoder_stochastic_depth_prob,
            style_decoder_stochastic_depth_prob=self.style_decoder_stochastic_depth_prob,
            style_encoder_norm_layer=self.style_encoder_norm_layer,
            style_decoder_norm_layer=self.style_decoder_norm_layer,
            style_encoder_MLP_activation_layer=self.style_encoder_MLP_activation_layer,
            style_decoder_MLP_activation_layer=self.style_decoder_MLP_activation_layer,
            style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation=self.style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation,
            style_decoder_use_instance_norm_with_affine=self.style_decoder_use_instance_norm_with_affine,
            style_decoder_use_regular_MHA_instead_of_Swin_at_the_end=self.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end,
            style_decoder_use_Key_instance_norm_after_linear_transformation=self.style_decoder_use_Key_instance_norm_after_linear_transformation,
            style_decoder_exclude_MLP_after_Fcs_self_MHA=self.style_decoder_exclude_MLP_after_Fcs_self_MHA,
            decoder_initializer=self.decoder_initializer
        )




        # Send models to device
        self.master_style_transformer.to(self.device)

        if(self.verbose):
            # Print network information
            self.print_network(self.master_style_transformer, 'StyleTransformer')



        # initialize the style transformer with truncated normal initialization
        for name, param in self.master_style_transformer.style_transformer.named_parameters():
            if 'weight' in name:
                torch.nn.init.trunc_normal_(param, std=0.02)
            elif 'bias' in name:
                torch.nn.init.constant_(param, 0)


        # set the devices to training mode
        self.master_style_transformer.style_transformer.train()
        self.master_style_transformer.decoder.train()
        if not self.freeze_encoder:
            self.master_style_transformer.swin_encoder.train()
        else:
            self.master_style_transformer.swin_encoder.eval()

            # Freeze the parameters of the encoder
            for param in self.master_style_transformer.swin_encoder.parameters():
                param.requires_grad = False

        # declare the image transform
        if self.use_imagenet_normalization_for_swin:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((512, 512)), # -> resize to 512x512
                transforms.RandomCrop((256,256)) , # random crop to 256x256
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # normalize with mean and std
            ])
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((512, 512)), # -> resize to 512x512
                transforms.RandomCrop((256,256)) , # random crop to 256x256
                transforms.ToTensor()
            ])

        
        # declare normalization for the loss function
        if self.use_imagenet_normalization_for_loss:
            self.imagenet_normalization_before_loss = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


        # Initialize loss function
        self.loss_function = custom_loss(project_absolute_path=self.project_root,
                                         feature_extractor_model_relative_path=self.loss_model_path,
                                         use_vgg19_with_batchnorm=self.use_vgg19_with_batchnorm,
                                         default_lambda_value=self.lambda_style,
                                         distance=self.loss_distance).to(self.device)




    def print_network(self, model, name):
        """Print out the network information."""
        num_params = 0
        for p in model.parameters():
            num_params += p.numel()


        for module in model.modules():
            print(module.__class__.__name__)
            for n, param in module.named_parameters():
                if param is not None:
                    print(f"  - {n}: {param.size()}")
            break
        print(f"Total number of parameters: {num_params}\n\n")

    def save_models(self, iter):
        """Save the models."""
        style_transformer_path = os.path.join(self.project_root, self.model_save_path, self.exp_name, f"{self.exp_name}_style_transformer_{iter}.pt")
        swin_encoder_path = os.path.join(self.project_root, self.model_save_path, self.exp_name, f"{self.exp_name}_swin_encoder_{iter}.pt")
        decoder_path = os.path.join(self.project_root, self.model_save_path, self.exp_name, f"{self.exp_name}_decoder_{iter}.pt")
        torch.save(self.master_style_transformer.style_transformer.state_dict(), style_transformer_path)
        torch.save(self.master_style_transformer.decoder.state_dict(), decoder_path)

        if not self.freeze_encoder:
            torch.save(self.master_style_transformer.swin_encoder.state_dict(), swin_encoder_path)

    def copy_model_to_omega(self):
        """
        Deepcopy the model parameters to omega_style_transformer and omega_decoder, and omega_encoder (if not frozen).
        (Obtain omega parameters from theta parameters before each inner loop.)
        """
        omega_style_transformer = deepcopy(self.master_style_transformer.style_transformer).to(self.device).train()
        omega_decoder = deepcopy(self.master_style_transformer.decoder).to(self.device).train()
        if not self.freeze_encoder:
            omega_encoder = deepcopy(self.master_style_transformer.swin_encoder).to(self.device).train()

            return omega_style_transformer, omega_decoder, omega_encoder

        return omega_style_transformer, omega_decoder


    def train(self):


        # Initialize wandb             
        if self.use_wandb:
            mode = 'online' if self.online else 'offline'
            kwargs = {'name': self.exp_name, 'project': 'master_v2', 'config': config,
                    'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
            wandb.init(**kwargs)
        else:
            mode = 'disabled'


            

        # create dataset objects
        coco_train_dataset_object = coco_train_dataset(
            project_absolute_path=self.project_root,
            transform=self.transform,                        
            coco_dataset_relative_path=self.coco_dataset_path
        )
        wikiart_dataset_object = wikiart_dataset(
            project_absolute_path=self.project_root,
            transform=self.transform,
            wikiart_dataset_relative_path=self.wikiart_dataset_path
        )

        # Initialize Dataloaders
        if not self.use_infinite_sampler:
            coco_dataloader = DataLoader(coco_train_dataset_object,
                                         batch_size=self.batch_size_content,
                                         shuffle=self.shuffle,
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory,
                                         drop_last=True)
            wikiart_dataloader = DataLoader(wikiart_dataset_object,
                                            batch_size=self.batch_size_style,
                                            shuffle=self.shuffle,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            drop_last=True)
        else:
            coco_dataloader = DataLoader(coco_train_dataset_object,
                                         batch_size=self.batch_size_content,
                                         num_workers=self.num_workers,
                                         pin_memory=self.pin_memory,
                                         sampler=InfiniteSampler(coco_train_dataset_object))
            wikiart_dataloader = DataLoader(wikiart_dataset_object,
                                            batch_size=self.batch_size_style,
                                            num_workers=self.num_workers,
                                            pin_memory=self.pin_memory,
                                            sampler=InfiniteSampler(wikiart_dataset_object))
        

        # create dataloader iterators
        coco_iterator = iter(coco_dataloader)
        wikiart_iterator = iter(wikiart_dataloader)

        
        # Create a new style_transformer and decoder objects, and load the parameters
        if not self.freeze_encoder:
            omega_style_transformer, omega_decoder, omega_encoder = self.copy_model_to_omega()
        else:
            omega_style_transformer, omega_decoder = self.copy_model_to_omega()

        # Set the new optimizer for inner loops
        if not self.freeze_encoder:
            inner_loop_optimizer = optim.Adam(list(omega_style_transformer.parameters()) + \
                                    list(omega_decoder.parameters()) + \
                                    list(omega_encoder.parameters()), lr=self.inner_lr)
        else:
            inner_loop_optimizer = optim.Adam(list(omega_style_transformer.parameters()) + list(omega_decoder.parameters()), lr=self.inner_lr)



        for iteration in tqdm(range(1, self.max_iterations + 1)):

            # print the iteration count if verbose is True
            if self.verbose:
                print(f"Iteration: {iteration:>10}/{self.max_iterations}")

            # Sample a style image
            style_image = (next(wikiart_iterator)).to(self.device)
            if (self.batch_size_content % self.batch_size_style) == 0:
                style_image_batch = style_image.repeat((self.batch_size_content // self.batch_size_style), 1, 1, 1)
            else:
                style_image_batch = torch.cat((style_image.repeat((self.batch_size_content // self.batch_size_style), 1, 1, 1),
                                              style_image[:self.batch_size_content % self.batch_size_style]),
                                              dim=0)
                

            # load the transformer and decoder from main weights
            omega_style_transformer.load_state_dict(self.master_style_transformer.style_transformer.state_dict())
            omega_decoder.load_state_dict(self.master_style_transformer.decoder.state_dict())
            if not self.freeze_encoder:
                omega_encoder.load_state_dict(self.master_style_transformer.swin_encoder.state_dict())




            for inner_loop_index in range(1, self.num_inner_updates+1):
                # Sample a batch of content images
                content_images = next(coco_iterator)
                content_images = content_images.to(self.device)
                # Randomly select the number of layers to use
                num_layers = random.randint(1, self.max_layers)

                # Encode the content and style images using the Swin Transformer
                if not self.freeze_encoder:
                    encoded_content = omega_encoder(content_images)
                    encoded_style = omega_encoder(style_image_batch)
                else:
                    encoded_content = self.master_style_transformer.swin_encoder(content_images)
                    encoded_style = self.master_style_transformer.swin_encoder(style_image_batch)


                # style transfer using the style transformer with omega parameters, not the self.style_transformer
                transformed_output = omega_style_transformer(encoded_content, encoded_style, num_layers)

                transformed_output = transformed_output.permute(0, 3, 1, 2)

                # Decode the transformed output with omega parameters, not the self.decoder
                decoded_output = omega_decoder(transformed_output)


                
                if self.use_imagenet_normalization_for_loss: # if needed, normalize the images

                    if self.use_imagenet_normalization_for_swin: # only normalize the decoder output
                        # Compute inner loss
                        total_loss, content_loss, style_loss = self.loss_function(content_images,
                                                                                  style_image_batch,
                                                                                  self.imagenet_normalization_before_loss(decoded_output),
                                                                                  output_content_and_style_loss=True)
                    else: # normalize everything
                        # Compute inner loss
                        total_loss, content_loss, style_loss = self.loss_function(self.imagenet_normalization_before_loss(content_images),
                                                                                  self.imagenet_normalization_before_loss(style_image_batch),
                                                                                  self.imagenet_normalization_before_loss(decoded_output),
                                                                                  output_content_and_style_loss=True)
                else:
                    # Compute inner loss
                    total_loss, content_loss, style_loss = self.loss_function(content_images,
                                                                              style_image_batch,
                                                                              decoded_output,
                                                                              output_content_and_style_loss=True)
             
                # Print the loss values if verbose is True
                if self.verbose:
                    print(f"Inner Loop {inner_loop_index:>5}/{self.num_inner_updates} - Total Loss: {total_loss:.2f}, Content Loss: {content_loss:.2f}, Style Loss: {style_loss:.2f}, Num Layers: {num_layers}")
                    

                # Backpropagation and optimization
                inner_loop_optimizer.zero_grad()
                total_loss.backward()
                inner_loop_optimizer.step()



            # Update theta parameters with omega parameters

            # Update the style transformer and decoder parameters
            for name, param in self.master_style_transformer.style_transformer.named_parameters():
                param.data += self.outer_lr * (omega_style_transformer.state_dict()[name] - param)

            # Update the encoder parameters if not frozen
            if not self.freeze_encoder:
                for name, param in self.master_style_transformer.swin_encoder.named_parameters():
                    param.data += self.outer_lr * (omega_encoder.state_dict()[name] - param)

            # Update the decoder parameters
            for name, param in self.master_style_transformer.decoder.named_parameters():
                param.data += self.outer_lr * (omega_decoder.state_dict()[name] - param)
            



            if iteration % self.save_every == 0:
                # Save model periodically
                self.save_models(iteration)

                if self.use_wandb:
                    # Log Iteration, Losses and Images
                    wandb.log({'total_loss': total_loss,
                               'content_loss': content_loss,
                               'style_loss': style_loss,
                               'content_image': [wandb.Image(content_images[0])],
                               'style_image': [wandb.Image(style_image)],
                               'stylized_image': [wandb.Image(decoded_output[0])]})
            else:
                if self.use_wandb:
                    # Log Iteration and Losses
                    wandb.log({'total_loss': total_loss,
                                'content_loss': content_loss,
                                'style_loss': style_loss})
                    

            # put some new lines for better readability if verbose is True
            if self.verbose:
                print("\n\n")




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Master Model')

    # project path 
    parser.add_argument('--project_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
                        help='The absolute path of the project root directory.')
    
    parser.add_argument('--model_save_path', type=str, default="exps/models",
                        help='Relative path to save the models.')


    
    # Dataset paths
    parser.add_argument('--coco_dataset_path', type=str, default="datasets/coco_train_dataset/train2017",
                        help='Relative path to the COCO dataset directory.')
    
    parser.add_argument('--wikiart_dataset_path', type=str, default="datasets/wikiart/**",
                        help='Relative path to the Wikiart dataset directory.')




    # Loss model path
    parser.add_argument('--loss_model_path', type=str, default="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                        help="Relative path to the pre-trained VGG19 model cut at the last layer of relu 5_1.")
    
    parser.add_argument('--use_vgg19_with_batchnorm', type=bool, default=False,
                        help="If true, use the pre-trained VGG19 model with batch normalization.")



    # DataLoader parameters
    parser.add_argument('--batch_size_style', type=int, default=1,
                        help='Batch size for the style datasets')
    
    parser.add_argument('--batch_size_content', type=int, default=4,
                        help='Batch size for the content dataset')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    parser.add_argument('--shuffle', default=True,
                        help='Whether to shuffle the dataset')
    
    parser.add_argument('--use_infinite_sampler', default=True,
                        help='Whether to use the InfiniteSampler (if used, shuffle will be neglected)')
    
    parser.add_argument('--pin_memory', default=True,
                        help='Whether to pin memory for faster data transfer to CUDA')



    # Hyperparameters
    parser.add_argument('--freeze_encoder', default=True,
                        help='Freeze the parameters of the model.')
    
    parser.add_argument('--inner_lr', type=float, default=0.0001,
                        help='Inner learning rate (delta)')
    
    parser.add_argument('--outer_lr', type=float, default=0.0001,
                        help='Outer learning rate (eta)')
    
    parser.add_argument('--num_inner_updates', type=int, default=1,
                        help='Number of inner updates (k)')
    
    parser.add_argument('--max_layers', type=int, default=4,
                        help='Maximal number of stacked layers (T)')
    
    parser.add_argument('--lambda_style', type=float, default=10.0,
                        help='Weighting term for style loss (lambda)')
    
    parser.add_argument('--loss_distance', type=str, default='euclidian',
                        help='Distance metric for the loss function')
    
    parser.add_argument('--use_imagenet_normalization_for_swin', type=bool, default=True,
                        help='Use ImageNet normalization for Swin Transformer')
    
    parser.add_argument('--use_imagenet_normalization_for_loss', type=bool, default=True,
                        help='Use ImageNet normalization for the loss function')
    
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save the model every n iterations')
    
    parser.add_argument('--max_iterations', type=int, default=15000,
                        help='Number of iterations to train the model.')



    # MasterStyleTransferModel parameters
    parser.add_argument('--swin_model_relative_path', type=str, default="weights/swin_B_first_2_stages.pt",
                        help='Relative path to the Swin Transformer model.')
    
    parser.add_argument('--swin_variant', type=str, default="swin_B",
                        help='Swin Transformer variant.')
    
    parser.add_argument('--style_encoder_dim', type=int, default=256,
                        help='Dimension of the encoder.')
    parser.add_argument('--style_decoder_dim', type=int, default=256,
                        help='Dimension of the decoder.')
    
    parser.add_argument('--style_encoder_num_heads', type=int, default=8,
                        help='Number of heads in the encoder.')
    parser.add_argument('--style_decoder_num_heads', type=int, default=8,
                        help='Number of heads in the decoder.')
    
    parser.add_argument('--style_encoder_window_size', type=list, default=[8, 8],
                        help='Window size of the encoder.')
    parser.add_argument('--style_decoder_window_size', type=list, default=[8, 8],
                        help='Window size of the decoder.')
    
    parser.add_argument('--style_encoder_shift_size', type=list, default=[4, 4],
                        help='Shift size of the encoder.')
    parser.add_argument('--style_decoder_shift_size', type=list, default=[4, 4],
                        help='Shift size of the decoder.')
    
    parser.add_argument('--style_encoder_mlp_ratio', type=float, default=4.0,
                        help='MLP ratio of the encoder.')
    parser.add_argument('--style_decoder_mlp_ratio', type=float, default=4.0,
                        help='MLP ratio of the decoder.')
    
    parser.add_argument('--style_encoder_dropout', type=float, default=0.0,
                        help='Dropout rate of the encoder.')
    parser.add_argument('--style_decoder_dropout', type=float, default=0.0,
                        help='Dropout rate of the decoder.')
    
    parser.add_argument('--style_encoder_attention_dropout', type=float, default=0.0,
                        help='Attention dropout rate of the encoder.')
    parser.add_argument('--style_decoder_attention_dropout', type=float, default=0.0,
                        help='Attention dropout rate of the decoder.')
    
    parser.add_argument('--style_encoder_qkv_bias', type=bool, default=True,
                        help='Whether to use bias in the QKV projection of the encoder.')
    parser.add_argument('--style_decoder_qkv_bias', type=bool, default=True,
                        help='Whether to use bias in the QKV projection of the decoder.')
    
    parser.add_argument('--style_encoder_proj_bias', type=bool, default=True,
                        help='Whether to use bias in the projection of the encoder.')
    parser.add_argument('--style_decoder_proj_bias', type=bool, default=True,
                        help='Whether to use bias in the projection of the decoder.')
    
    parser.add_argument('--style_encoder_stochastic_depth_prob', type=float, default=0.1,
                        help='Stochastic depth probability of the encoder.')
    parser.add_argument('--style_decoder_stochastic_depth_prob', type=float, default=0.1,
                        help='Stochastic depth probability of the decoder.')
    
    parser.add_argument('--style_encoder_norm_layer', type=callable, default=None,
                        help='Normalization layer of the encoder.')
    parser.add_argument('--style_decoder_norm_layer', type=callable, default=nn.LayerNorm,
                        help='Normalization layer of the decoder.')
    
    parser.add_argument('--style_encoder_MLP_activation_layer', type=callable, default=nn.GELU,
                        help='Activation layer of the MLP in the encoder.')
    parser.add_argument('--style_decoder_MLP_activation_layer', type=callable, default=nn.GELU,
                        help='Activation layer of the MLP in the decoder.')
    
    parser.add_argument('--style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation', type=bool, default=True,
                        help='Whether to use processed Key in Scale and Shift calculation of the encoder.')
    
    parser.add_argument('--style_decoder_use_instance_norm_with_affine', type=bool, default=False,
                        help='Whether to use instance normalization with affine in the decoder.')
    
    parser.add_argument('--style_decoder_use_regular_MHA_instead_of_Swin_at_the_end', type=bool, default=False,
                        help='Whether to use regular MHA instead of Swin at the end of the decoder.')
    
    parser.add_argument('--style_decoder_use_Key_instance_norm_after_linear_transformation', type=bool, default=True,
                        help='Whether to use instance normalization after linear transformation in the decoder.')
    
    parser.add_argument('--style_decoder_exclude_MLP_after_Fcs_self_MHA', type=bool, default=False,
                        help='Whether to exclude MLP after Fcs self MHA in the decoder.')
    
    parser.add_argument('--decoder_initializer', type=str, default="kaiming_normal_",
                        help='Initializer for the decoder.')



    # wandb configuration.
    parser.add_argument('--use_wandb', type=bool, default=False,
                        help='use wandb for logging')
    
    parser.add_argument('--online', type=bool, default=True,
                        help='use wandb online')
    
    parser.add_argument('--exp_name', type=str, default='master',
                        help='experiment name')




    # Seed configuration.
    parser.add_argument('--set_seed', type=bool, default=False,
                        help='set seed for reproducibility')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproducibility')



    # verbose
    parser.add_argument('--verbose', type=bool, default=True,
                        help='Print the model informations and loss values at each loss calculation.')


    config = parser.parse_args()
    train = Train(config)
    train.train()
