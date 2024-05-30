from typing import Any, Callable, List, Optional


import torch
import torch.nn.functional as F
from torch import nn, Tensor

import os
import sys
# add the project path to the system path
project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_absolute_path)

# import the function to download the swin model and create the cutted model
from codes.utils import download_swin_and_create_cutted_model
from codes.style_transformer import StyleTransformer
from codes.decoder import Decoder
from codes.load_pretrained_weights_to_style_transformer import get_pretained_weight_loaded_style_transformer_state_dict


class MasterStyleTransferModel(nn.Module):
    def __init__(self,
        project_absolute_path: str = project_absolute_path,
        swin_model_relative_path: str = None,
        swin_variant: str = "swin_B",
        style_encoder_dim: int = 256,
        style_decoder_dim: int = 256,
        style_encoder_num_heads: int = 8,
        style_decoder_num_heads: int = 8,
        style_encoder_window_size: List[int] = [8, 8],
        style_decoder_window_size: List[int] = [8, 8],
        style_encoder_shift_size: List[int] = [4, 4],
        style_decoder_shift_size: List[int] = [4, 4],
        style_encoder_mlp_ratio: float = 4.0,
        style_decoder_mlp_ratio: float = 4.0,
        style_encoder_dropout: float = 0.0,
        style_decoder_dropout: float = 0.0,
        style_encoder_attention_dropout: float = 0.0,
        style_decoder_attention_dropout: float = 0.0,
        style_encoder_qkv_bias: bool = True,
        style_decoder_qkv_bias: bool = True,
        style_encoder_proj_bias: bool = True,
        style_decoder_proj_bias: bool = True,
        style_encoder_stochastic_depth_prob: float = 0.1,
        style_decoder_stochastic_depth_prob: float = 0.1,
        style_encoder_norm_layer: Callable[..., nn.Module] = None,
        style_decoder_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        style_encoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        style_decoder_MLP_activation_layer: Optional[nn.Module] = nn.GELU,
        style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation: bool = True,
        style_decoder_use_instance_norm_with_affine: bool = False,
        style_decoder_use_regular_MHA_instead_of_Swin_at_the_end: bool = False,
        style_decoder_use_Key_instance_norm_after_linear_transformation: bool = True,
        style_decoder_exclude_MLP_after_Fcs_self_MHA: bool = False,
        style_transformer_load_pretrained_weights: bool = False,
        style_transformer_pretrained_weights_path: str = None,
        decoder_initializer: str = "kaiming_normal_",
    ):
        super(MasterStyleTransferModel, self).__init__()

        # download the model and save it
        download_swin_and_create_cutted_model(absolute_project_path = project_absolute_path,
                                            model_save_relative_path = swin_model_relative_path,
                                            swin_variant = swin_variant)
        
        # load the model
        self.swin_encoder = torch.load(os.path.join(project_absolute_path, swin_model_relative_path))



        self.style_transformer = StyleTransformer(
            encoder_dim = style_encoder_dim,
            decoder_dim = style_decoder_dim,
            encoder_num_heads = style_encoder_num_heads,
            decoder_num_heads = style_decoder_num_heads,
            encoder_window_size = style_encoder_window_size,
            decoder_window_size = style_decoder_window_size,
            encoder_shift_size = style_encoder_shift_size,
            decoder_shift_size = style_decoder_shift_size,
            encoder_mlp_ratio = style_encoder_mlp_ratio,
            decoder_mlp_ratio = style_decoder_mlp_ratio,
            encoder_dropout = style_encoder_dropout,
            decoder_dropout = style_decoder_dropout,
            encoder_attention_dropout = style_encoder_attention_dropout,
            decoder_attention_dropout = style_decoder_attention_dropout,
            encoder_qkv_bias = style_encoder_qkv_bias,
            decoder_qkv_bias = style_decoder_qkv_bias,
            encoder_proj_bias = style_encoder_proj_bias,
            decoder_proj_bias = style_decoder_proj_bias,
            encoder_stochastic_depth_prob = style_encoder_stochastic_depth_prob,
            decoder_stochastic_depth_prob = style_decoder_stochastic_depth_prob,
            encoder_norm_layer = style_encoder_norm_layer,
            decoder_norm_layer = style_decoder_norm_layer,
            encoder_MLP_activation_layer = style_encoder_MLP_activation_layer,
            decoder_MLP_activation_layer = style_decoder_MLP_activation_layer,
            encoder_if_use_processed_Key_in_Scale_and_Shift_calculation = style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation,
            decoder_use_instance_norm_with_affine = style_decoder_use_instance_norm_with_affine,
            decoder_use_regular_MHA_instead_of_Swin_at_the_end = style_decoder_use_regular_MHA_instead_of_Swin_at_the_end,
            decoder_use_Key_instance_norm_after_linear_transformation = style_decoder_use_Key_instance_norm_after_linear_transformation,
            decoder_exclude_MLP_after_Fcs_self_MHA = style_decoder_exclude_MLP_after_Fcs_self_MHA,
        )
        self.style_encoder_dim = style_encoder_dim
        self.style_decoder_dim = style_decoder_dim
        self.style_encoder_num_heads = style_encoder_num_heads
        self.style_decoder_num_heads = style_decoder_num_heads
        self.style_encoder_window_size = style_encoder_window_size
        self.style_decoder_window_size = style_decoder_window_size
        self.style_encoder_shift_size = style_encoder_shift_size
        self.style_decoder_shift_size = style_decoder_shift_size
        self.style_encoder_mlp_ratio = style_encoder_mlp_ratio
        self.style_decoder_mlp_ratio = style_decoder_mlp_ratio
        self.style_encoder_dropout = style_encoder_dropout
        self.style_decoder_dropout = style_decoder_dropout
        self.style_encoder_attention_dropout = style_encoder_attention_dropout
        self.style_decoder_attention_dropout = style_decoder_attention_dropout
        self.style_encoder_qkv_bias = style_encoder_qkv_bias
        self.style_decoder_qkv_bias = style_decoder_qkv_bias
        self.style_encoder_proj_bias = style_encoder_proj_bias
        self.style_decoder_proj_bias = style_decoder_proj_bias
        self.style_encoder_stochastic_depth_prob = style_encoder_stochastic_depth_prob
        self.style_decoder_stochastic_depth_prob = style_decoder_stochastic_depth_prob
        self.style_encoder_norm_layer = style_encoder_norm_layer
        self.style_decoder_norm_layer = style_decoder_norm_layer
        self.style_encoder_MLP_activation_layer = style_encoder_MLP_activation_layer
        self.style_decoder_MLP_activation_layer = style_decoder_MLP_activation_layer
        self.style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation = style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation
        self.style_decoder_use_instance_norm_with_affine = style_decoder_use_instance_norm_with_affine
        self.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end = style_decoder_use_regular_MHA_instead_of_Swin_at_the_end
        self.style_decoder_use_Key_instance_norm_after_linear_transformation = style_decoder_use_Key_instance_norm_after_linear_transformation
        self.style_decoder_exclude_MLP_after_Fcs_self_MHA = style_decoder_exclude_MLP_after_Fcs_self_MHA
        self.style_transformer_load_pretrained_weights = style_transformer_load_pretrained_weights
        self.style_transformer_pretrained_weights_path = style_transformer_pretrained_weights_path

        self.decoder = Decoder(channel_dim=style_decoder_dim,
                               initializer=decoder_initializer)
        
        if style_transformer_load_pretrained_weights:
            self.load_pretained_weights_to_style_transformer(pretrained_weights_path=style_transformer_pretrained_weights_path)
        
    

    def load_pretained_weights_to_style_transformer(self, pretrained_weights_path: str):
        if pretrained_weights_path is None:
            raise ValueError("Please provide the path of the pretrained weights")
        
        if self.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end:
            raise ValueError("The pretrained weights are not compatible with the current model configuration. Please set the style_decoder_use_regular_MHA_instead_of_Swin_at_the_end to False")
        
        # get the models state_dict
        state_dict = self.style_transformer.state_dict()

        # deep copy the state_dict
        state_dict_save = {key: value.clone() for key, value in state_dict.items()}

        # load the pretrained weights
        pretrained_state_dict = get_pretained_weight_loaded_style_transformer_state_dict(
            state_dict = state_dict,
            shifted_window_block_path = self.style_transformer_pretrained_weights_path,
            encoder_dim = self.style_encoder_dim,
            decoder_dim = self.style_decoder_dim,
            encoder_mlp_ratio = self.style_decoder_mlp_ratio,
            decoder_mlp_ratio = self.style_decoder_mlp_ratio,
            encoder_window_size = self.style_encoder_window_size,
            decoder_window_size = self.style_decoder_window_size,
            encoder_qkv_bias = self.style_encoder_qkv_bias,
            decoder_qkv_bias = self.style_decoder_qkv_bias,
            encoder_proj_bias = self.style_encoder_proj_bias, 
            decoder_proj_bias = self.style_decoder_proj_bias,
            encoder_norm_layer = self.style_encoder_norm_layer, 
            decoder_norm_layer = self.style_decoder_norm_layer,
            decoder_use_regular_MHA_instead_of_Swin_at_the_end = self.style_decoder_use_regular_MHA_instead_of_Swin_at_the_end,
            decoder_exclude_MLP_after_Fcs_self_MHA = self.style_decoder_exclude_MLP_after_Fcs_self_MHA,
        )

        # load the pretrained weights to the model
        self.style_transformer.load_state_dict(pretrained_state_dict)

        if_loaded_correctly = True

        # check if the weights are loaded correctly (all the weight should be changed)
        for key in state_dict.keys():
            if "relative_position" in key: # ignore the relative position embeddings (dimensions are different)
                continue

            if torch.equal(state_dict_save[key], pretrained_state_dict[key]):
                print(f"\n\n\nPRETRAINED WEIGHTS ARE NOT LOADED CORRECTLY FOR KEY: {key}\n\n\n")
                if_loaded_correctly = False

        if if_loaded_correctly:
            print("\n\n\nPRETRAINED WEIGHTS ARE LOADED CORRECTLY!\n\n\n")
        else:
            print("\n\n\nPRETRAINED WEIGHTS ARE NOT LOADED CORRECTLY!\n\n\n")


        

    def forward(self,
                content_image: Tensor,
                style_image: Tensor,
                transformer_layer_count: int = 1) -> Tensor:
        
        content_image = self.swin_encoder(content_image)
        style_image = self.swin_encoder(style_image)

        Fcs = self.style_transformer(content_image, style_image, transformer_layer_count).permute(0, 3, 1, 2)

        Ics = self.decoder(Fcs)
        
        return Ics
    


if __name__ == "__main__":

    # test the model

    # declare the current relative path for the swin model
    swin_model_relative_path = os.path.join("weights", "swin_B_first_2_stages.pt")

    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    # create the model
    model = MasterStyleTransferModel(project_absolute_path=project_absolute_path,
                                     swin_model_relative_path=swin_model_relative_path,
                                     swin_variant="swin_B",
                                     style_encoder_dim=256,
                                     style_decoder_dim=256,
                                     style_encoder_num_heads=8,
                                     style_decoder_num_heads=8,
                                     style_encoder_window_size=[7, 7],
                                     style_decoder_window_size=[7, 7],
                                     style_encoder_shift_size=[4, 4],
                                     style_decoder_shift_size=[4, 4],
                                     style_encoder_mlp_ratio=4.0,
                                     style_decoder_mlp_ratio=4.0,
                                     style_encoder_dropout=0.0,
                                     style_decoder_dropout=0.0,
                                     style_encoder_attention_dropout=0.0,
                                     style_decoder_attention_dropout=0.0,
                                     style_encoder_qkv_bias=True,
                                     style_decoder_qkv_bias=True,
                                     style_encoder_proj_bias=True,
                                     style_decoder_proj_bias=True,
                                     style_encoder_stochastic_depth_prob=0.1,
                                     style_decoder_stochastic_depth_prob=0.1,
                                     style_encoder_norm_layer=None,
                                     style_decoder_norm_layer=nn.LayerNorm,
                                     style_encoder_MLP_activation_layer=nn.GELU,
                                     style_decoder_MLP_activation_layer=nn.GELU,
                                     style_encoder_if_use_processed_Key_in_Scale_and_Shift_calculation=True,
                                     style_decoder_use_instance_norm_with_affine=False,
                                     style_decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
                                     style_decoder_use_Key_instance_norm_after_linear_transformation=True,
                                     style_transformer_load_pretrained_weights=True,
                                     style_transformer_pretrained_weights_path=os.path.join(project_absolute_path, "weights", "model_basic_layer_1_module_list_shifted_window_block_state_dict.pth"),
                                     decoder_initializer="default"
                                     )
    
    # set the model to evaluation mode
    model.eval()


    # print model summary
    # print(model)

    # print model total parameter count
    print(f"Model total parameter count: {sum(p.numel() for p in model.parameters())}")


    # create a dummy input

    content_image = torch.randn(1, 3, 256, 256)
    style_image = torch.randn(1, 3, 256, 256)

    # normalize the input
    content_image = content_image / 255.0
    style_image = style_image / 255.0

    # normalize with imagenet statistics
    content_image[:, 0] = (content_image[:, 0] - 0.485) / 0.229
    content_image[:, 1] = (content_image[:, 1] - 0.456) / 0.224
    content_image[:, 2] = (content_image[:, 2] - 0.406) / 0.225

    # get the output

    output = model(content_image, style_image, transformer_layer_count=1)


    # denormalize the output
    output[:, 0] = (output[:, 0] * 0.229) + 0.485
    output[:, 1] = (output[:, 1] * 0.224) + 0.456
    output[:, 2] = (output[:, 2] * 0.225) + 0.406


    print(f"Input shape: {content_image.shape}")
    print(f"Output shape: {output.shape}")

    
    print(output)
    print(f"Output shape: {output.shape}")

                 