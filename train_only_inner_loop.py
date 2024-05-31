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
        self.warmup_epochs = config.warmup_epochs
        self.decay_lr_until = config.decay_lr_until
        self.decay_lr_rate = config.decay_lr_rate
        self.decay_every = config.decay_every
        self.max_layers = config.max_layers
        self.lambda_style = config.lambda_style
        self.loss_distance_content = config.loss_distance_content
        self.loss_distance_style = config.loss_distance_style
        self.use_random_crop = config.use_random_crop
        self.use_imagenet_normalization_for_swin = config.use_imagenet_normalization_for_swin
        self.use_imagenet_normalization_for_loss = config.use_imagenet_normalization_for_loss
        self.save_every = config.save_every
        self.save_every_for_model = config.save_every_for_model
        self.max_iterations = config.max_iterations

        self.fast_adaptation_stage_on = config.fast_adaptation_stage_on
        self.pretrained_style_transformer_path = config.pretrained_style_transformer_path
        self.pretrained_decoder_path = config.pretrained_decoder_path




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
        self.style_transformer_load_pretrained_weights = config.style_transformer_load_pretrained_weights
        self.style_transformer_pretrained_weights_path = config.style_transformer_pretrained_weights_path



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

        
        # check if the fast adaptation stage is on
        if self.fast_adaptation_stage_on:
            # check if paths are given
            if(self.pretrained_style_transformer_path == ''):
                raise ValueError("Pre-trained style transformer path is not given!")
            if(self.pretrained_decoder_path == ''):
                raise ValueError("Pre-trained decoder path is not given!")


        # Initialize the master style transfer model
        with torch.no_grad():
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
                style_transformer_load_pretrained_weights=self.style_transformer_load_pretrained_weights,
                style_transformer_pretrained_weights_path=self.style_transformer_pretrained_weights_path,
                decoder_initializer=self.decoder_initializer,
                direct_pretrained_style_transformer_path=self.pretrained_style_transformer_path,
                direct_pretrained_decoder_path=self.pretrained_decoder_path
            )


        if((not self.style_transformer_load_pretrained_weights) and (not self.fast_adaptation_stage_on)):
            print("\nInitializing the weights of the style transformer with truncated normal initialization!\n")
            self.master_style_transformer.apply(self._init_weights_style_transformer)

        self.master_style_transformer.train()




        # Send models to device
        self.master_style_transformer.to(self.device)

        if(self.verbose):
            # Print network information
            self.print_network(self.master_style_transformer, 'StyleTransformer')




        if self.freeze_encoder:
            # Freeze the parameters of the encoder
            for param in self.master_style_transformer.swin_encoder.parameters():
                param.requires_grad = False


        # declare the image transform
        if self.use_random_crop:
            if self.verbose:
                print("Using random crop for the images!")
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((512, 512)), # -> resize to 512x512
                transforms.RandomCrop((256,256)) , # random crop to 256x256
                transforms.ToTensor()
            ])
        else:
            if self.verbose:
                print("Using center crop for the images!")
            self.transform = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((512, 512)), # -> resize to 512x512
                transforms.CenterCrop((256,256)) , # center crop to 256x256
                transforms.ToTensor()
            ])
        
        if self.fast_adaptation_stage_on:
            self.transform_style_for_fast_adaptation = transforms.Compose([
                transforms.ToPILImage(), # -> PIL image
                transforms.Resize((512, 512)), # -> resize to 512x512
                transforms.CenterCrop((256,256)) , # center crop to 256x256
                transforms.ToTensor()
            ])




        # declare normalization for the loss function
        if self.use_imagenet_normalization_for_loss or self.use_imagenet_normalization_for_swin:
            self.imagenet_normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



        # Initialize loss function
        self.loss_function = custom_loss(project_absolute_path=self.project_root,
                                         feature_extractor_model_relative_path=self.loss_model_path,
                                         use_vgg19_with_batchnorm=self.use_vgg19_with_batchnorm,
                                         default_lambda_value=self.lambda_style,
                                         distance_content=self.loss_distance_content,
                                         distance_style=self.loss_distance_style).to(self.device)


        if self.fast_adaptation_stage_on:
            # freeze everything except the style encoder
            for param in self.master_style_transformer.swin_encoder.parameters():
                param.requires_grad = False

            for param in self.master_style_transformer.style_transformer.decoder.parameters():
                param.requires_grad = False
            
            for param in self.master_style_transformer.style_transformer.encoder.parameters():
                param.requires_grad = True
            
            for param in self.master_style_transformer.decoder.parameters():
                param.requires_grad = False

            self.lr_schedule_on = False
        else:
            if (self.warmup_epochs or self.decay_lr_until):
                if (self.warmup_epochs and self.decay_lr_until):
                    self.lr_schedule_on = True
                else:
                    print("Please provide both warmup epochs and decay learning rate until epoch!")
                    self.lr_schedule_on = False
            else:
                self.lr_schedule_on = False
            
        if self.verbose:
            if self.lr_schedule_on:
                print(f"Using learning rate scheduling with warmup epochs: {self.warmup_epochs} and decay learning rate until lr value: {self.decay_lr_until}!")
            else:
                print("Not using learning rate scheduling!")




    def schedule_lr(self, optimizer, iteration):
        """Decay the learning rate."""
        if iteration <= self.warmup_epochs:
            # increase linearly, starting from 1/100 of the initial learning rate (iteration starts from 1)
            lr = self.inner_lr * (((iteration) / (self.warmup_epochs))*0.99 + 0.01)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                return lr
        else:
            # decay exponentially every decay_every iterations
            if iteration % self.decay_every == 0:
                # calculate the new learning rate
                lr = self.inner_lr * (self.decay_lr_rate ** (iteration // self.decay_every))
                lr = max(lr, self.decay_lr_until)

                # set the new learning rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                return lr



    # Initialize the weights of the model (style transformer part)
    def _init_weights_style_transformer(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)



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



    def train(self):


        # Initialize wandb             
        if self.use_wandb:
            mode = 'online' if self.online else 'offline'
            kwargs = {'name': self.exp_name, 'project': 'master_v2', 'config': config,
                    'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
            wandb.init(**kwargs)
        else:
            mode = 'disabled'


        if self.set_seed:
            np.random.seed(self.seed)
            random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

            os.environ["PYTHONHASHSEED"] = str(self.seed)

        # create dataset objects
        coco_train_dataset_object = coco_train_dataset(
            project_absolute_path=self.project_root,
            transform=self.transform,                        
            coco_dataset_relative_path=self.coco_dataset_path
        )

        if not self.fast_adaptation_stage_on:
            wikiart_dataset_object = wikiart_dataset(
                project_absolute_path=self.project_root,
                transform=self.transform,
                wikiart_dataset_relative_path=self.wikiart_dataset_path
            )
        else:
            wikiart_dataset_object = wikiart_dataset(
                project_absolute_path=self.project_root,
                transform=self.transform_style_for_fast_adaptation,
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

        

        # Set the new optimizer for inner loops
        if not self.freeze_encoder:
            optimizer = optim.Adam([ 
                                {'params': self.master_style_transformer.style_transformer.parameters()},
                                {'params': self.master_style_transformer.decoder.parameters()},
                                ], lr=self.inner_lr)
        else:
            optimizer = optim.Adam([ 
                                {'params': self.master_style_transformer.swin_encoder.parameters()},
                                {'params': self.master_style_transformer.style_transformer.parameters()},
                                {'params': self.master_style_transformer.decoder.parameters()},
                                ], lr=self.inner_lr)


        for iteration in tqdm(range(1, self.max_iterations + 1)):

            # print the iteration count if verbose is True
            if self.verbose:
                print(f"Iteration: {iteration:>10}/{self.max_iterations}")


            # Sample a style image
            style_image = (next(wikiart_iterator))

            if (self.batch_size_content % self.batch_size_style) == 0:
                style_image_batch = style_image.repeat((self.batch_size_content // self.batch_size_style), 1, 1, 1)
            else:
                style_image_batch = torch.cat((style_image.repeat((self.batch_size_content // self.batch_size_style), 1, 1, 1),
                                              style_image[:self.batch_size_content % self.batch_size_style]),
                                              dim=0)
                
            if (self.use_imagenet_normalization_for_swin and (not self.use_imagenet_normalization_for_loss)):
                style_image_batch_non_normalized = style_image_batch.clone()
                style_image_batch = self.imagenet_normalization(style_image_batch)
            elif (self.use_imagenet_normalization_for_swin):
                style_image_batch = self.imagenet_normalization(style_image_batch)
                
            style_image_batch = style_image_batch.to(self.device)



            # Sample a batch of content images
            content_images = next(coco_iterator)

            if (self.use_imagenet_normalization_for_swin and (not self.use_imagenet_normalization_for_loss)):
                content_images_non_normalized = content_images.clone()
                content_images = self.imagenet_normalization(content_images)
            elif (self.use_imagenet_normalization_for_swin):
                content_images = self.imagenet_normalization(content_images)

            content_images = content_images.to(self.device)
            # Randomly select the number of layers to use
            num_layers = random.randint(1, self.max_layers)


            # style transfer using the style transformer with omega parameters, not the self.style_transformer
            decoded_output = self.master_style_transformer(content_images, style_image_batch, num_layers)
            



            
            if self.use_imagenet_normalization_for_loss: # if needed, normalize the images
                if self.use_imagenet_normalization_for_swin: # only normalize the decoder output
                    # Compute inner loss
                    total_loss, content_loss, style_loss = self.loss_function(content_images,
                                                                                style_image_batch,
                                                                                self.imagenet_normalization(decoded_output),
                                                                                output_content_and_style_loss=True)
                else: # normalize everything
                    # Compute inner loss
                    total_loss, content_loss, style_loss = self.loss_function(self.imagenet_normalization(content_images),
                                                                                self.imagenet_normalization(style_image_batch),
                                                                                self.imagenet_normalization(decoded_output),
                                                                                output_content_and_style_loss=True)
            else:
                if self.use_imagenet_normalization_for_swin:
                    content_images = content_images.to("cpu")
                    style_image_batch = style_image_batch.to("cpu")

                    content_images_non_normalized = content_images_non_normalized.to(self.device)
                    style_image_batch_non_normalized = style_image_batch_non_normalized.to(self.device)



                    # Compute inner loss
                    total_loss, content_loss, style_loss = self.loss_function(content_images_non_normalized,
                                                                              style_image_batch_non_normalized,
                                                                              decoded_output,
                                                                              output_content_and_style_loss=True)
                    
                    content_images_non_normalized = content_images_non_normalized.to("cpu")
                    style_image_batch_non_normalized = style_image_batch_non_normalized.to("cpu")
                else:
                    # Compute inner loss
                    total_loss, content_loss, style_loss = self.loss_function(content_images,
                                                                                style_image_batch,
                                                                                decoded_output,
                                                                                output_content_and_style_loss=True)
            
            # Print the loss values if verbose is True
            if self.verbose:
                print(f"Total Loss: {total_loss:.2f}, Content Loss: {content_loss:.2f}, Style Loss: {style_loss:.2f}, Num Layers: {num_layers}")
                

            # Backpropagation and optimization
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()


            # Decay the learning rate
            if self.lr_schedule_on:
                self.schedule_lr(optimizer, iteration)


            if iteration % self.save_every == 0:
                if self.use_wandb:
                    # Log Iteration, Losses and Images
                    wandb.log({'total_loss': total_loss,
                               'content_loss': content_loss,
                               'style_loss': style_loss,
                               'learning_rate': optimizer.param_groups[0]['lr'],
                               'content_image': [wandb.Image(content_images[0])],
                               'style_image': [wandb.Image(style_image)],
                               'stylized_image': [wandb.Image(decoded_output[0])]})
            else:
                if self.use_wandb:
                    # Log Iteration and Losses
                    wandb.log({'total_loss': total_loss,
                                'content_loss': content_loss,
                                'style_loss': style_loss,
                                'learning_rate': optimizer.param_groups[0]['lr']})
                    
            if iteration % self.save_every_for_model == 0:
                # Save model periodically
                self.save_models(iteration)
                    

            # put some new lines for better readability if verbose is True
            if self.verbose:
                print("\n\n")




if __name__ == '__main__':

    # define str2bool function for argparse
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    def str2listint(v):
        # strip the string and split it by comma
        v = v.strip().split(',')
        # convert the string to integer
        v = [int(i) for i in v]
        return v

        

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
    
    parser.add_argument('--use_vgg19_with_batchnorm', type=str2bool, nargs='?', const=True, default=False,
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
    
    parser.add_argument('--warmup_epochs', type=int, default=0,
                        help='Number of warmup epochs')
    
    parser.add_argument('--decay_lr_until', type=float, default=0.,
                        help='Decay learning rate until the given epoch')
    
    parser.add_argument('--decay_lr_rate', type=float, default=0.1,
                        help='Decay learning rate by the given rate')
    
    parser.add_argument('--decay_every', type=int, default=1000,
                        help='Decay learning rate every n iterations')
    
    parser.add_argument('--max_layers', type=int, default=4,
                        help='Maximal number of stacked layers (T)')
    
    parser.add_argument('--lambda_style', type=float, default=10.0,
                        help='Weighting term for style loss (lambda)')
    
    parser.add_argument('--loss_distance_content', type=str, default='euclidian',
                        help='Distance metric for the loss function')
    
    parser.add_argument('--loss_distance_style', type=str, default='euclidian',
                        help='Distance metric for the loss function')
                        
    
    parser.add_argument('--use_random_crop', type=str2bool, nargs='?', const=True, default=True,
                        help='Use random crop for the images (if False, use center crop)')
    
    parser.add_argument('--use_imagenet_normalization_for_swin', type=str2bool, nargs='?', const=True, default=True,
                        help='Use ImageNet normalization for Swin Transformer')
    
    parser.add_argument('--use_imagenet_normalization_for_loss', type=str2bool, nargs='?', const=True, default=True,
                        help='Use ImageNet normalization for the loss function')
    
    parser.add_argument('--save_every', type=int, default=100,
                        help='Save the model every n iterations')
    
    parser.add_argument('--save_every_for_model', type=int, default=1000,
                        help='Save the model every n iterations')
    
    parser.add_argument('--max_iterations', type=int, default=15000,
                        help='Number of iterations to train the model.')
    
    parser.add_argument('--fast_adaptation_stage_on', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to use fast adaptation stage')
    
    parser.add_argument('--pretrained_style_transformer_path', type=str, default='',
                        help='Path to the pre-trained style transformer for fast adaptation stage')
    
    parser.add_argument('--pretrained_decoder_path', type=str, default='',
                        help='Path to the pre-trained decoder for fast adaptation stage')



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
    
    parser.add_argument('--style_encoder_window_size', type=str2listint, nargs='?', const=True, default=[7, 7],
                        help='Window size of the encoder.')
    parser.add_argument('--style_decoder_window_size', type=str2listint, nargs='?', const=True, default=[7, 7],
                        help='Window size of the decoder.')
    
    parser.add_argument('--style_encoder_shift_size', type=str2listint, nargs='?', const=True, default=[4, 4],
                        help='Shift size of the encoder.')
    parser.add_argument('--style_decoder_shift_size', type=str2listint, nargs='?', const=True, default=[4, 4],
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
    
    parser.add_argument('--style_encoder_qkv_bias', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to use bias in the QKV projection of the encoder.')
    parser.add_argument('--style_decoder_qkv_bias', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to use bias in the QKV projection of the decoder.')
    
    parser.add_argument('--style_encoder_proj_bias', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to use bias in the projection of the encoder.')
    parser.add_argument('--style_decoder_proj_bias', type=str2bool, nargs='?', const=True, default=True,
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
    
    parser.add_argument('--style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to use processed Key in Scale and Shift calculation of the encoder.')
    
    parser.add_argument('--style_decoder_use_instance_norm_with_affine', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to use instance normalization with affine in the decoder.')
    
    parser.add_argument('--style_decoder_use_regular_MHA_instead_of_Swin_at_the_end', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to use regular MHA instead of Swin at the end of the decoder.')
    
    parser.add_argument('--style_decoder_use_Key_instance_norm_after_linear_transformation', type=str2bool, nargs='?', const=True, default=True,
                        help='Whether to use instance normalization after linear transformation in the decoder.')
    
    parser.add_argument('--style_decoder_exclude_MLP_after_Fcs_self_MHA', type=str2bool, nargs='?', const=True, default=False,
                        help='Whether to exclude MLP after Fcs self MHA in the decoder.')
    
    parser.add_argument('--decoder_initializer', type=str, default="kaiming_normal_",
                        help='Initializer for the decoder.')
    
    parser.add_argument('--style_transformer_load_pretrained_weights', type=str2bool, nargs='?', const=True, default=False,
                        help='Load the pre-trained weights for the style transformer (from an original swin block).')
    
    parser.add_argument('--style_transformer_pretrained_weights_path', type=str, default="weights/model_basic_layer_1_module_list_shifted_window_block_state_dict.pth",
                        help='Relative path to the pre-trained weights for the style transformer.')



    # wandb configuration.
    parser.add_argument('--use_wandb', type=str2bool, nargs='?', const=True, default=False,
                        help='use wandb for logging')
    
    parser.add_argument('--online', type=str2bool, nargs='?', const=True, default=True,
                        help='use wandb online')
    
    parser.add_argument('--exp_name', type=str, default='master',
                        help='experiment name')




    # Seed configuration.
    parser.add_argument('--set_seed', type=str2bool, nargs='?', const=True, default=False,
                        help='set seed for reproducibility')
    
    parser.add_argument('--seed', type=int, default=42,
                        help='seed for reproducibility')



    # verbose
    parser.add_argument('--verbose', type=str2bool, nargs='?', const=True, default=True,
                        help='Print the model informations and loss values at each loss calculation.')


    config = parser.parse_args()



    train = Train(config)
    train.train()