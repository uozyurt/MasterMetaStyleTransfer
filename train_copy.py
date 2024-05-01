import argparse
import os
import random
from tqdm import tqdm
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb

from codes.models import SwinEncoder, StyleTransformer, StyleDecoder
from codes.loss import custom_loss
from codes.get_dataloader import coco_train_dataset, wikiart_dataset


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

        # Paths
        self.project_root = config.project_root
        #self.coco_dataset_path = os.path.join(self.project_root, config.coco_dataset_path)
        #self.wikiart_dataset_path = os.path.join(self.project_root, config.wikiart_dataset_path)
        #self.encoder_model_path = os.path.join(self.project_root, config.encoder_model_path)
        #self.loss_model_path = os.path.join(self.project_root, config.loss_model_path)
        self.model_save_path = config.model_save_path
        self.coco_dataset_path = config.coco_dataset_path
        self.wikiart_dataset_path = config.wikiart_dataset_path
        self.encoder_model_path = config.encoder_model_path
        self.loss_model_path = config.loss_model_path

        # Dataloader parameters
        self.batch_size_style = config.batch_size_style
        self.batch_size_content = config.batch_size_content
        self.num_workers = config.num_workers
        self.shuffle = config.shuffle
        self.pin_memory = config.pin_memory

        # Model parameters
        self.dim = config.dim
        self.input_resolution = config.input_resolution
        self.num_heads = config.num_heads
        self.window_size = config.window_size
        self.shift_size = config.shift_size
        self.mlp_ratio = config.mlp_ratio
        self.qkv_bias = config.qkv_bias
        self.qk_scale = config.qk_scale
        self.drop = config.drop
        self.attn_drop = config.attn_drop
        self.act_layer = config.act_layer
        self.freeze_encoder = config.freeze_encoder

        # Hyperparameters
        self.inner_lr = config.inner_lr
        self.outer_lr = config.outer_lr
        self.num_inner_updates = config.num_inner_updates
        self.max_layers = config.max_layers
        self.lambda_style = config.lambda_style
        self.save_every = config.save_every
        self.max_iterations = config.max_iterations

        # Initialize models
        self.style_transformer = StyleTransformer(
            dim=self.dim,
            input_resolution=tuple(self.input_resolution),
            num_heads=self.num_heads,
            window_size=self.window_size,
            shift_size=self.shift_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=self.qkv_bias,
            qk_scale=self.qk_scale,
            drop=self.drop,
            attn_drop=self.attn_drop
        )

        self.swin_encoder = SwinEncoder(
            relative_model_path=self.encoder_model_path,
            freeze_params=self.freeze_encoder
        )

        self.decoder = StyleDecoder()

        # Send models to device
        self.style_transformer.to(self.device)
        self.swin_encoder.to(self.device)
        self.decoder.to(self.device)

        # Print network information
        self.print_network(self.style_transformer, 'StyleTransformer')
        self.print_network(self.swin_encoder, 'SwinEncoder')
        self.print_network(self.decoder, 'Decoder')

        """        # Initialize datasets
        coco_dataset = coco_train_dataset(project_absolute_path=self.project_root, coco_dataset_relative_path=self.coco_dataset_path)
        wikiart_dataset = wikiart_dataset(project_absolute_path=self.project_root, wikiart_dataset_relative_path=self.wikiart_dataset_path)

        # Initialize dataloaders
        coco_dataloader = DataLoader(coco_dataset, batch_size=self.batch_size_content, shuffle=self.shuffle,
                                    num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)
        wikiart_dataloader = DataLoader(wikiart_dataset, batch_size=self.batch_size_style, shuffle=self.shuffle,
                                        num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=self.drop_last)"""

        # Initialize optimizer here
        if self.freeze_encoder:
            self.parameters = list(self.style_transformer.parameters()) + list(self.decoder.parameters())
        else:
            self.parameters = list(self.style_transformer.parameters()) + list(self.swin_encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.Adam(self.parameters, lr=self.outer_lr)

        # Initialize loss function
        self.loss_function = custom_loss(self.project_root, self.loss_model_path, self.lambda_style).to(self.device)

        # Wandb parameters
        self.use_wandb = config.use_wandb
        self.online = config.online
        self.exp_name = config.exp_name

        # Seed configuration
        self.set_seed = config.set_seed
        self.seed = config.seed

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
        style_transformer_path = os.path.join(self.project_root, self.model_save_path, f"{self.exp_name}_style_transformer_{iter}.pt")
        swin_encoder_path = os.path.join(self.project_root, self.model_save_path, f"{self.exp_name}_swin_encoder_{iter}.pt")
        decoder_path = os.path.join(self.project_root, self.model_save_path, f"{self.exp_name}_decoder_{iter}.pt")
        torch.save(self.style_transformer.state_dict(), style_transformer_path)
        torch.save(self.decoder.state_dict(), decoder_path)

        if not self.freeze_encoder:
            torch.save(self.swin_encoder.state_dict(), swin_encoder_path)


    # def copy_model_to_omega(self):
    #     """Deepcopy the model parameters to omega_style_transformer, omega_decoder, and omega_encoder (if not frozen)."""
    #     omega_style_transformer = StyleTransformer(
    #         dim=self.dim,
    #         input_resolution=tuple(self.input_resolution),
    #         num_heads=self.num_heads,
    #         window_size=self.window_size,
    #         shift_size=self.shift_size,
    #         mlp_ratio=self.mlp_ratio,
    #         qkv_bias=self.qkv_bias,
    #         qk_scale=self.qk_scale,
    #         drop=self.drop,
    #         attn_drop=self.attn_drop
    #     ).to(self.device)
    #     omega_style_transformer.load_state_dict(self.style_transformer.state_dict())

    #     omega_decoder = StyleDecoder().to(self.device)
    #     omega_decoder.load_state_dict(self.decoder.state_dict())

    #     if not self.freeze_encoder:
    #         omega_encoder = SwinEncoder(
    #             relative_model_path=self.encoder_model_path,
    #             freeze_params=self.freeze_encoder
    #         ).to(self.device)
    #         omega_encoder.load_state_dict(self.swin_encoder.state_dict())

    #     return omega_style_transformer, omega_decoder, omega_encoder


    def copy_model_to_omega(self):
        """Deepcopy the model parameters to omega_style_transformer and omega_decoder, and omega_encoder (if not frozen)."""
        omega_style_transformer = deepcopy(self.style_transformer).to(self.device).train()
        omega_decoder = deepcopy(self.decoder).to(self.device).train()
        if not self.freeze_encoder:
            omega_encoder = deepcopy(self.swin_encoder).to(self.device).train()

            return omega_style_transformer, omega_decoder, omega_encoder

        return omega_style_transformer, omega_decoder


    def train(self):
        if self.use_wandb:
            mode = 'online' if self.online else 'offline'
        else:
            mode = 'disabled'
        kwargs = {'name': self.exp_name, 'project': 'master', 'config': config,
                  'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode, 'save_code': True}
        wandb.init(**kwargs)

        # Make sure model saving path exists
        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Initialize Dataloaders
        coco_dataloader = DataLoader(coco_train_dataset(self.project_root, self.coco_dataset_path),
                                     batch_size=self.batch_size_content, shuffle=self.shuffle,
                                     num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)
        wikiart_dataloader = DataLoader(wikiart_dataset(self.project_root, self.wikiart_dataset_path),
                                        batch_size=self.batch_size_style, shuffle=self.shuffle,
                                        num_workers=self.num_workers, pin_memory=self.pin_memory, drop_last=True)

        total_loss_prev, content_loss_prev, style_loss_prev = 0, 0, 0

        for iteration in tqdm(range(1, self.max_iterations + 1)):
            wandb.log({'iteration': iteration})
            # Sample a style image
            style_image = next(iter(wikiart_dataloader))
            style_image = style_image.to(self.device)
            style_image_batch = style_image.repeat(self.batch_size_content, 1, 1, 1)


            
            # Create a new style_transformer and decoder objects, and load the parameters
            if not self.freeze_encoder:
                omega_style_transformer, omega_decoder, omega_encoder = self.copy_model_to_omega()
            else:
                omega_style_transformer, omega_decoder = self.copy_model_to_omega()

            # Set the new optimizer for inner loops
            optimizer = optim.Adam(list(omega_style_transformer.parameters()) + list(omega_decoder.parameters()), lr=self.inner_lr)



            for _ in range(self.num_inner_updates):
                # Sample a batch of content images
                content_images = next(iter(coco_dataloader))
                content_images = content_images.to(self.device)
                # Randomly select the number of layers to use
                num_layers = random.randint(1, self.max_layers)

                # Encode the content and style images using the Swin Transformer
                if not self.freeze_encoder:
                    encoded_content = self.swin_encoder(content_images)
                    encoded_style = self.swin_encoder(style_image_batch)
                else:
                    encoded_content = self.swin_encoder(content_images)
                    encoded_style = self.swin_encoder(style_image_batch)


                # style transfer using the style transformer with omega parameters, not the self.style_transformer
                transformed_output = omega_style_transformer(encoded_content, encoded_style, num_layers)

                transformed_output = transformed_output.permute(0, 3, 1, 2)

                # Decode the transformed output with omega parameters, not the self.decoder
                decoded_output = omega_decoder(transformed_output)

                # Compute inner loss
                total_loss, content_loss, style_loss = self.loss_function(content_images, style_image_batch, decoded_output, output_content_and_style_loss=True)

                # Print losses with their differences
                print(f"Total loss:   {total_loss.item():.2f} diff:({total_loss.item() - total_loss_prev:.2f})")
                print(f"Content loss: {content_loss.item():.2f} diff:({content_loss.item() - content_loss_prev:.2f})")
                print(f"Style loss:   {style_loss.item():.2f} diff:({style_loss.item() - style_loss_prev:.2f})")
                print()

                total_loss_prev, content_loss_prev, style_loss_prev = total_loss.item(), content_loss.item(), style_loss.item()

                # Log losses
                wandb.log({'total_loss': total_loss})
                wandb.log({'content_loss': content_loss})
                wandb.log({'style_loss': style_loss})
                # Backpropagation and optimization
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()



            # Update theta parameters with omega parameters

            # Update the style transformer and decoder parameters
            for name, param in self.style_transformer.named_parameters():
                param.data += self.outer_lr * (omega_style_transformer.state_dict()[name] - param)

            # Update the encoder parameters if not frozen
            if not self.freeze_encoder:
                for name, param in self.swin_encoder.named_parameters():
                    param.data += self.outer_lr * (omega_encoder.state_dict()[name] - param)

            # Update the decoder parameters
            for name, param in self.decoder.named_parameters():
                param.data += self.outer_lr * (omega_decoder.state_dict()[name] - param)
            
            # Save model periodically
            if iteration % self.save_every == 0:
                self.save_models(iteration)

                # Log sample images
                wandb.log({'content_image': [wandb.Image(content_images[0])]})
                wandb.log({'style_image': [wandb.Image(style_image)]})
                wandb.log({'stylized_image': [wandb.Image(decoded_output[0])]})



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Master Model')

    # project path 
    parser.add_argument('--project_root', type=str, default=os.path.abspath(os.path.join(os.path.dirname(__file__), ".")),
                        help='The absolute path of the project root directory.')
    parser.add_argument('--model_save_path', type=str, default="exps/models",)

    # Loss model path
    parser.add_argument('--loss_model_path', type=str, default="weights/vgg_19_last_layer_is_relu_5_1_output.pt",
                        help="Relative path to the pre-trained VGG19 model cut at the last layer of relu 5_1.")

    # StyleTransformer parameters
    parser.add_argument('--dim', type=int, default=256, help='Number of input channels.')
    parser.add_argument('--input_resolution', type=int, nargs=2, default=[32, 32], help='Dimensions (height, width) of the input feature map.')
    parser.add_argument('--num_heads', type=int, default=8, help='Number of attention heads.')
    parser.add_argument('--window_size', type=int, default=8, help='Size of the attention window.')
    parser.add_argument('--shift_size', type=int, default=4, help='Offset for the cyclic shift within the window attention mechanism.')
    parser.add_argument('--mlp_ratio', type=float, default=4.0, help='Expansion ratio for the MLP block compared to the number of input channels.')
    parser.add_argument('--qkv_bias', type=bool, default=True, help='Add a learnable bias to query, key, value projections.')
    parser.add_argument('--qk_scale', type=float, default=None, help='Custom scaling factor for query-key dot products in attention.')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate applied to the output of the MLP block.')
    parser.add_argument('--attn_drop', type=float, default=0.0, help='Dropout rate applied to attention weights.')
    parser.add_argument('--act_layer', type=str, default='nn.ReLU', help='Activation function used in the MLP block.')

    # SwinEncoder parameters
    parser.add_argument('--encoder_model_path', type=str, default='weights/swin_B_first_2_stages.pt', help='Path where the Swin model is saved or should be saved.')
    parser.add_argument('--freeze_encoder', default=True, help='Freeze the parameters of the model.')

    # Hyperparameters
    parser.add_argument('--inner_lr', type=float, default=0.0001, help='Inner learning rate (delta)')
    parser.add_argument('--outer_lr', type=float, default=0.0001, help='Outer learning rate (eta)')
    parser.add_argument('--num_inner_updates', type=int, default=4, help='Number of inner updates (k)')
    parser.add_argument('--max_layers', type=int, default=4, help='Maximal number of stacked layers (T)')
    parser.add_argument('--lambda_style', type=float, default=10.0, help='Weighting term for style loss (lambda)')
    parser.add_argument('--save_every', type=int, default=100, help='Save the model every n iterations')
    # Dataset paths
    parser.add_argument('--coco_dataset_path', type=str, default="datasets/coco_train_dataset/train2017",
                        help='Relative path to the COCO dataset directory.')
    parser.add_argument('--wikiart_dataset_path', type=str, default="datasets/wikiart/**",
                        help='Relative path to the Wikiart dataset directory.')
    parser.add_argument('--max_iterations', type=int, default=20000, help='Number of iterations to train the model.')

    # DataLoader parameters
    parser.add_argument('--batch_size_style', type=int, default=1, help='Batch size for the style datasets')
    parser.add_argument('--batch_size_content', type=int, default=4, help='Batch size for the content dataset')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading')
    parser.add_argument('--shuffle', action='store_true', help='Whether to shuffle the dataset')
    parser.add_argument('--pin_memory', action='store_true', help='Whether to pin memory for faster data transfer to CUDA')

    # Seed configuration.
    parser.add_argument('--set_seed', type=bool, default=False, help='set seed for reproducibility')
    parser.add_argument('--seed', type=int, default=42, help='seed for reproducibility')

    # wandb configuration.
    parser.add_argument('--use_wandb', type=bool, default=False, help='use wandb for logging')
    parser.add_argument('--online', type=bool, default=True, help='use wandb online')
    parser.add_argument('--exp_name', type=str, default='master', help='experiment name')


    config = parser.parse_args()
    train = Train(config)
    train.train()