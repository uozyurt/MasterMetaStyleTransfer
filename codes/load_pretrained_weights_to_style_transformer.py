import sys
import os
from collections import OrderedDict
from typing import Callable, List, Optional

# add the project path to the sys path
project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append("/home/umut/projects/796_project")


import torch
import torch.nn as nn



class PretrainedSwinWeightsLoader:
    def __init__(self,shifted_window_block_path = "weights/model_basic_layer_1_module_list_shifted_window_block_state_dict.pth"):
        
        # cutted part should be obtained from pretrained swin_base_patch4_window7_224_22kto1k model (original swin transformer by microsoft) or an equivalent implementation
        # the weights are specifically cutted from the shifted window block in the exact location from the original full model:
        # ModuleList -> 2. BasicLayer -> ModuleList -> 2. SwinTransformerBlock
        

        # open the state dict
        pretrained_state_dict = torch.load(shifted_window_block_path)


        # assign the values to the variables (get torch.nn.parameter.Parameter insted of tensors)
        self.linear_norm1_weight = pretrained_state_dict["0.weight"].cpu()
        self.linear_norm1_bias = pretrained_state_dict["0.bias"].cpu()


        self.relative_position_bias_table = pretrained_state_dict["1.relative_position_bias_table"].cpu()
        self.relative_position_index = pretrained_state_dict["1.relative_position_index"].cpu().flatten()

        qkv_weight = pretrained_state_dict["1.qkv.weight"].cpu()
        qkv_bias = pretrained_state_dict["1.qkv.bias"].cpu()

        self.proj_weight = pretrained_state_dict["1.proj.weight"].cpu()
        self.proj_bias = pretrained_state_dict["1.proj.bias"].cpu()

        self.layer_norm_2_weight = pretrained_state_dict["3.weight"].cpu()
        self.layer_norm_2_bias = pretrained_state_dict["3.bias"].cpu()

        self.mlp_layer1_weight = pretrained_state_dict["4.fc1.weight"].cpu()
        self.mlp_layer1_bias = pretrained_state_dict["4.fc1.bias"].cpu()

        self.mlp_layer2_weight = pretrained_state_dict["4.fc2.weight"].cpu()
        self.mlp_layer2_bias = pretrained_state_dict["4.fc2.bias"].cpu()


        # seperate qkv weight to q, k, v
        self.q_weight = qkv_weight[:qkv_weight.shape[0]//3]
        self.k_weight = qkv_weight[qkv_weight.shape[0]//3:2*qkv_weight.shape[0]//3]
        self.v_weight = qkv_weight[2*qkv_weight.shape[0]//3:]

        # seperate qkv bias to q, k, v
        self.q_bias = qkv_bias[:qkv_bias.shape[0]//3]
        self.k_bias = qkv_bias[qkv_bias.shape[0]//3:2*qkv_bias.shape[0]//3]
        self.v_bias = qkv_bias[2*qkv_bias.shape[0]//3:]



    # laod weights to the style transformer (state dict of the model should be obtained before calling this function)
    def load_weights_to_style_transformer(self,
        style_transformer: OrderedDict,
        encoder_dim: int,
        decoder_dim: int,
        encoder_mlp_ratio: int,
        decoder_mlp_ratio: int,
        encoder_window_size: List[int],
        decoder_window_size: List[int],
        encoder_qkv_bias: bool = True,
        decoder_qkv_bias: bool = True,
        encoder_proj_bias: bool = True,
        decoder_proj_bias: bool = True,
        encoder_norm_layer: Callable[..., nn.Module] = None,
        decoder_norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end: bool = False,
        decoder_exclude_MLP_after_Fcs_self_MHA: bool = False,
    ):
        # check if the dimensions and ratios are correct
        assert encoder_dim == 256, "encoder_dim should be 256 for pre-trained weight loading"
        assert decoder_dim == 256, "decoder_dim should be 256 for pre-trained weight loading"

        assert encoder_mlp_ratio == 4, "encoder_mlp_ratio should be 4 for pre-trained weight loading"
        assert decoder_mlp_ratio == 4, "decoder_mlp_ratio should be 4 for pre-trained weight loading"

        assert encoder_window_size == [7, 7], "encoder_window_size should be [7, 7] for pre-trained weight loading"
        assert decoder_window_size == [7, 7], "decoder_window_size should be [7, 7] for pre-trained weight loading"

        # load the weights to the encoder
        state_dict = self.load_encoder(
            state_dict = style_transformer,
            encoder_qkv_bias = encoder_qkv_bias,
            encoder_proj_bias = encoder_proj_bias,
            encoder_norm_layer = encoder_norm_layer,
            encoder_exclude_MLP_after = True,
        )

        # load the weights to the decoder
        state_dict = self.load_decoder(
            state_dict = state_dict,
            decoder_qkv_bias = decoder_qkv_bias,
            decoder_proj_bias = decoder_proj_bias,
            decoder_norm_layer = decoder_norm_layer,
            decoder_exclude_MLP_after_Fcs_self_MHA = decoder_exclude_MLP_after_Fcs_self_MHA,
        )

        return state_dict


    def load_encoder(
        self,
        state_dict,
        encoder_qkv_bias,
        encoder_proj_bias,
        encoder_norm_layer,
        encoder_exclude_MLP_after = True,
    ):
        state_dict = self.load_encoder_shared_MHA_without_MLP(
            state_dict = state_dict,
            encoder_qkv_bias = encoder_qkv_bias,
            encoder_proj_bias = encoder_proj_bias,
            encoder_norm_layer = encoder_norm_layer,
            encoder_exclude_MLP_after = encoder_exclude_MLP_after,
        )

        state_dict = self.load_encoder_MLP_Key(
            state_dict = state_dict,
        )

        state_dict = self.load_encoder_MLP_Scale(
            state_dict = state_dict,
        )

        state_dict = self.load_encoder_MLP_Shift(
            state_dict = state_dict,
        )

        return state_dict


    def load_decoder(
        self,
        state_dict,
        decoder_qkv_bias,
        decoder_proj_bias,
        decoder_norm_layer,
        decoder_exclude_MLP_after_Fcs_self_MHA,
    ):
        state_dict = self.load_decoder_MHA_self_attn(
            state_dict = state_dict,
            decoder_qkv_bias = decoder_qkv_bias,
            decoder_proj_bias = decoder_proj_bias,
            decoder_norm_layer = decoder_norm_layer,
            decoder_exclude_MLP_after_Fcs_self_MHA = decoder_exclude_MLP_after_Fcs_self_MHA,
        )

        state_dict = self.decoder_last_MLP(
            state_dict = state_dict,
        )

        state_dict = self.decoder_MHA_for_sigma_and_mu(
            state_dict = state_dict,
            decoder_qkv_bias = decoder_qkv_bias,
            decoder_proj_bias = decoder_proj_bias,
            use_q_proj = False,
        )

        return state_dict


    def load_encoder_shared_MHA_without_MLP(
        self,
        state_dict,
        encoder_qkv_bias,
        encoder_proj_bias,
        encoder_norm_layer = None,
        encoder_exclude_MLP_after = True,
    ):
        if encoder_qkv_bias:
            q_bias_name = "encoder.shared_MHA_without_MLP.attn.Wq.bias"
            k_bias_name = "encoder.shared_MHA_without_MLP.attn.Wk.bias"
            v_bias_name = "encoder.shared_MHA_without_MLP.attn.Wv.bias"
        else:
            q_bias_name = None
            k_bias_name = None
            v_bias_name = None
        
        if encoder_proj_bias:
            proj_bias_name = "encoder.shared_MHA_without_MLP.attn.proj.bias"
        else:
            proj_bias_name = None

        if encoder_norm_layer:
            linear_norm1_weight_name = "encoder.shared_MHA_without_MLP.norm1.weight"
            linear_norm1_bias_name = "encoder.shared_MHA_without_MLP.norm1.bias"
            linear_norm2_weight_name = "encoder.shared_MHA_without_MLP.norm2.weight"
            linear_norm2_bias_name = "encoder.shared_MHA_without_MLP.norm2.bias"
        else:
            linear_norm1_weight_name = None
            linear_norm1_bias_name = None
            linear_norm2_weight_name = None
            linear_norm2_bias_name = None
            

        if not encoder_exclude_MLP_after:
            mlp_layer1_weight_name = "encoder.shared_MHA_without_MLP.mlp.0.weight"
            mlp_layer1_bias_name = "encoder.shared_MHA_without_MLP.mlp.0.bias"
            mlp_layer2_weight_name = "encoder.shared_MHA_without_MLP.mlp.3.weight"
            mlp_layer2_bias_name = "encoder.shared_MHA_without_MLP.mlp.3.bias"
        else:
            mlp_layer1_weight_name = None
            mlp_layer1_bias_name = None
            mlp_layer2_weight_name = None
            mlp_layer2_bias_name = None
            
    
        return self.load_weights(
            state_dict = state_dict,
            linear_norm1_weight = linear_norm1_weight_name,
            linear_norm1_bias = linear_norm1_bias_name,
            relative_position_bias_table = "encoder.shared_MHA_without_MLP.attn.relative_position_bias_table",
            relative_position_index = "encoder.shared_MHA_without_MLP.attn.relative_position_index",
            q_weight = "encoder.shared_MHA_without_MLP.attn.Wq.weight",
            k_weight = "encoder.shared_MHA_without_MLP.attn.Wk.weight",
            v_weight = "encoder.shared_MHA_without_MLP.attn.Wv.weight",
            q_bias = q_bias_name,
            k_bias = k_bias_name,
            v_bias = v_bias_name,
            proj_weight = "encoder.shared_MHA_without_MLP.attn.proj.weight",
            proj_bias = proj_bias_name,
            layer_norm_2_weight = linear_norm2_weight_name,
            layer_norm_2_bias = linear_norm2_bias_name,
            mlp_layer1_weight = mlp_layer1_weight_name,
            mlp_layer1_bias = mlp_layer1_bias_name,
            mlp_layer2_weight = mlp_layer2_weight_name,
            mlp_layer2_bias = mlp_layer2_bias_name,
        )


    def load_encoder_MLP_Key(
        self,
        state_dict,
    ):
        
        return self.load_weights(
            state_dict = state_dict,
            mlp_layer1_weight = "encoder.encoder_MLP_Key.0.weight",
            mlp_layer1_bias = "encoder.encoder_MLP_Key.0.bias",
            mlp_layer2_weight = "encoder.encoder_MLP_Key.3.weight",
            mlp_layer2_bias = "encoder.encoder_MLP_Key.3.bias",
        )


    def load_encoder_MLP_Scale(
        self,
        state_dict,
    ):
            
        return self.load_weights(
            state_dict = state_dict,
            mlp_layer1_weight = "encoder.encoder_MLP_Scale.0.weight",
            mlp_layer1_bias = "encoder.encoder_MLP_Scale.0.bias",
            mlp_layer2_weight = "encoder.encoder_MLP_Scale.3.weight",
            mlp_layer2_bias = "encoder.encoder_MLP_Scale.3.bias",
        )


    def load_encoder_MLP_Shift(
        self,
        state_dict,
    ):
        
        return self.load_weights(
            state_dict = state_dict,
            mlp_layer1_weight = "encoder.encoder_MLP_Shift.0.weight",
            mlp_layer1_bias = "encoder.encoder_MLP_Shift.0.bias",
            mlp_layer2_weight = "encoder.encoder_MLP_Shift.3.weight",
            mlp_layer2_bias = "encoder.encoder_MLP_Shift.3.bias",
        )

    
    def load_decoder_MHA_self_attn(
        self,
        state_dict,
        decoder_qkv_bias,
        decoder_proj_bias,
        decoder_norm_layer,
        decoder_exclude_MLP_after_Fcs_self_MHA = False,
    ):
        if decoder_qkv_bias:
            q_bias_name = "decoder.MHA_self_attn.attn.Wq.bias"
            k_bias_name = "decoder.MHA_self_attn.attn.Wk.bias"
            v_bias_name = "decoder.MHA_self_attn.attn.Wv.bias"
        else:
            q_bias_name = None
            k_bias_name = None
            v_bias_name = None
        
        if decoder_proj_bias:
            proj_bias_name = "decoder.MHA_self_attn.attn.proj.bias"
        else:
            proj_bias_name = None

        if decoder_norm_layer:

            linear_norm1_weight_name = "decoder.MHA_self_attn.norm1.weight"
            linear_norm1_bias_name = "decoder.MHA_self_attn.norm1.bias"

            linear_norm2_weight_name = "decoder.MHA_self_attn.norm2.weight"
            linear_norm2_bias_name = "decoder.MHA_self_attn.norm2.bias"


        if not decoder_exclude_MLP_after_Fcs_self_MHA:
            mlp_layer1_weight_name = "decoder.MHA_self_attn.mlp.0.weight"
            mlp_layer1_bias_name = "decoder.MHA_self_attn.mlp.0.bias"
            mlp_layer2_weight_name = "decoder.MHA_self_attn.mlp.3.weight"
            mlp_layer2_bias_name = "decoder.MHA_self_attn.mlp.3.bias"
        else:
            mlp_layer1_weight_name = None
            mlp_layer1_bias_name = None
            mlp_layer2_weight_name = None
            mlp_layer2_bias_name = None
            
    
        return self.load_weights(
            state_dict = state_dict,
            linear_norm1_weight = linear_norm1_weight_name,
            linear_norm1_bias = linear_norm1_bias_name,
            relative_position_bias_table = "decoder.MHA_self_attn.attn.relative_position_bias_table",
            relative_position_index = "decoder.MHA_self_attn.attn.relative_position_index",
            q_weight = "decoder.MHA_self_attn.attn.Wq.weight",
            k_weight = "decoder.MHA_self_attn.attn.Wk.weight",
            v_weight = "decoder.MHA_self_attn.attn.Wv.weight",
            q_bias = q_bias_name,
            k_bias = k_bias_name,
            v_bias = v_bias_name,
            proj_weight = "decoder.MHA_self_attn.attn.proj.weight",
            proj_bias = proj_bias_name,
            layer_norm_2_weight = linear_norm2_weight_name,
            layer_norm_2_bias = linear_norm2_bias_name,
            mlp_layer1_weight = mlp_layer1_weight_name,
            mlp_layer1_bias = mlp_layer1_bias_name,
            mlp_layer2_weight = mlp_layer2_weight_name,
            mlp_layer2_bias = mlp_layer2_bias_name,
        )


    def decoder_last_MLP(
        self,
        state_dict,
    ):
        
        return self.load_weights(
            state_dict = state_dict,
            mlp_layer1_weight = "decoder.last_MLP.0.weight",
            mlp_layer1_bias = "decoder.last_MLP.0.bias",
            mlp_layer2_weight = "decoder.last_MLP.3.weight",
            mlp_layer2_bias = "decoder.last_MLP.3.bias",
        )


    def decoder_MHA_for_sigma_and_mu(
        self,
        state_dict,
        decoder_qkv_bias,
        decoder_proj_bias,
        use_q_proj = False,
    ): 
        if decoder_qkv_bias:
            q_bias_name = "decoder.decoder_MHA_for_sigma_and_mu.Wk.bias"
            k_bias_name = "decoder.decoder_MHA_for_sigma_and_mu.Wk.bias"
            v_scale_bias_name = "decoder.decoder_MHA_for_sigma_and_mu.Wv_scale.bias"
            v_shift_bias_name = "decoder.decoder_MHA_for_sigma_and_mu.Wv_shift.bias"
        else:
            q_bias_name = None
            k_bias_name = None
            v_scale_bias_name = None
            v_shift_bias_name = None
        
        if decoder_proj_bias:
            proj_bias_name = "decoder.decoder_MHA_for_sigma_and_mu.proj.bias"
        else:
            proj_bias_name = None

        if use_q_proj:
            q_weight_name = "decoder.decoder_MHA_for_sigma_and_mu.Wq.weight"
        else:
            q_weight_name = None
            q_bias_name = None

        return self.load_weights(
            state_dict = state_dict,
            relative_position_bias_table = "decoder.decoder_MHA_for_sigma_and_mu.relative_position_bias_table",
            relative_position_index = "decoder.decoder_MHA_for_sigma_and_mu.relative_position_index",
            q_weight = q_weight_name,
            k_weight = "decoder.decoder_MHA_for_sigma_and_mu.Wk.weight",
            q_bias = q_bias_name,
            k_bias = k_bias_name,
            v_scale_weight= "decoder.decoder_MHA_for_sigma_and_mu.Wv_scale.weight",
            v_shift_weight = "decoder.decoder_MHA_for_sigma_and_mu.Wv_shift.weight",
            v_scale_bias = v_scale_bias_name,
            v_shift_bias = v_shift_bias_name,
            proj_weight = "decoder.decoder_MHA_for_sigma_and_mu.proj.weight",
            proj_bias = proj_bias_name,
        )
    
        

        

            
    def load_weights(
            self,
            state_dict: OrderedDict,
            linear_norm1_weight: str = False,
            linear_norm1_bias: str = False,
            relative_position_bias_table: str = False,
            relative_position_index: str = False,
            q_weight: str = False,
            k_weight: str = False,
            v_weight: str = False,
            q_bias: str = False,
            k_bias: str = False,
            v_bias: str = False,
            proj_weight: str = False,
            proj_bias: str = False,
            layer_norm_2_weight: str = False,
            layer_norm_2_bias: str = False,
            mlp_layer1_weight: str = False,
            mlp_layer1_bias: str = False,
            mlp_layer2_weight: str = False,
            mlp_layer2_bias: str = False,
            v_scale_weight: str = False,
            v_shift_weight: str = False,
            v_scale_bias: str = False,
            v_shift_bias: str = False,
    ):
        if linear_norm1_weight:
            # check if the shapes and types are correct
            if(state_dict[linear_norm1_weight].shape != self.linear_norm1_weight.shape):
                print(f"\n\n\nshape mismatch for {linear_norm1_weight} (original: {state_dict[linear_norm1_weight].shape}, new: {self.linear_norm1_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[linear_norm1_weight].dtype != self.linear_norm1_weight.dtype):
                print(f"\n\n\ndtype mismatch for {linear_norm1_weight} (original: {state_dict[linear_norm1_weight].dtype}, new: {self.linear_norm1_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[linear_norm1_weight] = self.linear_norm1_weight
        
        if linear_norm1_bias:
            # check if the shapes and types are correct
            if(state_dict[linear_norm1_bias].shape != self.linear_norm1_bias.shape):
                print(f"\n\n\nshape mismatch for {linear_norm1_bias} (original: {state_dict[linear_norm1_bias].shape}, new: {self.linear_norm1_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[linear_norm1_bias].dtype != self.linear_norm1_bias.dtype):
                print(f"\n\n\ndtype mismatch for {linear_norm1_bias} (original: {state_dict[linear_norm1_bias].dtype}, new: {self.linear_norm1_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[linear_norm1_bias] = self.linear_norm1_bias

        if relative_position_bias_table:
            # check if the shapes and types are correct
            if(state_dict[relative_position_bias_table].shape != self.relative_position_bias_table.shape):
                print(f"\n\n\nshape mismatch for {relative_position_bias_table} (original: {state_dict[relative_position_bias_table].shape}, new: {self.relative_position_bias_table.shape})\n\n\n")
                raise ValueError
            if(state_dict[relative_position_bias_table].dtype != self.relative_position_bias_table.dtype):
                print(f"\n\n\ndtype mismatch for {relative_position_bias_table} (original: {state_dict[relative_position_bias_table].dtype}, new: {self.relative_position_bias_table.dtype})\n\n\n")
                raise ValueError

            state_dict[relative_position_bias_table] = self.relative_position_bias_table
            
        if relative_position_index:
            # check if the shapes and types are correct
            if(state_dict[relative_position_index].shape != self.relative_position_index.shape):
                print(f"\n\n\nshape mismatch for {relative_position_index} (original: {state_dict[relative_position_index].shape}, new: {self.relative_position_index.shape})\n\n\n")
                raise ValueError
            if(state_dict[relative_position_index].dtype != self.relative_position_index.dtype):
                print(f"\n\n\ndtype mismatch for {relative_position_index} (original: {state_dict[relative_position_index].dtype}, new: {self.relative_position_index.dtype})\n\n\n")
                raise ValueError

            state_dict[relative_position_index] = self.relative_position_index

        if q_weight:
            # check if the shapes and types are correct
            if(state_dict[q_weight].shape != self.q_weight.shape):
                print(f"\n\n\nshape mismatch for {q_weight} (original: {state_dict[q_weight].shape}, new: {self.q_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[q_weight].dtype != self.q_weight.dtype):
                print(f"\n\n\ndtype mismatch for {q_weight} (original: {state_dict[q_weight].dtype}, new: {self.q_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[q_weight] = self.q_weight

        if k_weight:
            # check if the shapes and types are correct
            if(state_dict[k_weight].shape != self.k_weight.shape):
                print(f"\n\n\nshape mismatch for {k_weight} (original: {state_dict[k_weight].shape}, new: {self.k_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[k_weight].dtype != self.k_weight.dtype):
                print(f"\n\n\ndtype mismatch for {k_weight} (original: {state_dict[k_weight].dtype}, new: {self.k_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[k_weight] = self.k_weight

        if v_weight:
            # check if the shapes and types are correct
            if(state_dict[v_weight].shape != self.v_weight.shape):
                print(f"\n\n\nshape mismatch for {v_weight} (original: {state_dict[v_weight].shape}, new: {self.v_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_weight].dtype != self.v_weight.dtype):
                print(f"\n\n\ndtype mismatch for {v_weight} (original: {state_dict[v_weight].dtype}, new: {self.v_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[v_weight] = self.v_weight

        if q_bias:
            # check if the shapes and types are correct
            if(state_dict[q_bias].shape != self.q_bias.shape):
                print(f"\n\n\nshape mismatch for {q_bias} (original: {state_dict[q_bias].shape}, new: {self.q_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[q_bias].dtype != self.q_bias.dtype):
                print(f"\n\n\ndtype mismatch for {q_bias} (original: {state_dict[q_bias].dtype}, new: {self.q_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[q_bias] = self.q_bias

        if k_bias:
            # check if the shapes and types are correct
            if(state_dict[k_bias].shape != self.k_bias.shape):
                print(f"\n\n\nshape mismatch for {k_bias} (original: {state_dict[k_bias].shape}, new: {self.k_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[k_bias].dtype != self.k_bias.dtype):
                print(f"\n\n\ndtype mismatch for {k_bias} (original: {state_dict[k_bias].dtype}, new: {self.k_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[k_bias] = self.k_bias

        if v_bias:
            # check if the shapes and types are correct
            if(state_dict[v_bias].shape != self.v_bias.shape):
                print(f"\n\n\nshape mismatch for {v_bias} (original: {state_dict[v_bias].shape}, new: {self.v_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_bias].dtype != self.v_bias.dtype):
                print(f"\n\n\ndtype mismatch for {v_bias} (original: {state_dict[v_bias].dtype}, new: {self.v_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[v_bias] = self.v_bias

        if proj_weight:
            # check if the shapes and types are correct
            if(state_dict[proj_weight].shape != self.proj_weight.shape):
                print(f"\n\n\nshape mismatch for {proj_weight} (original: {state_dict[proj_weight].shape}, new: {self.proj_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[proj_weight].dtype != self.proj_weight.dtype):
                print(f"\n\n\ndtype mismatch for {proj_weight} (original: {state_dict[proj_weight].dtype}, new: {self.proj_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[proj_weight] = self.proj_weight

        if proj_bias:
            # check if the shapes and types are correct
            if(state_dict[proj_bias].shape != self.proj_bias.shape):
                print(f"\n\n\nshape mismatch for {proj_bias} (original: {state_dict[proj_bias].shape}, new: {self.proj_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[proj_bias].dtype != self.proj_bias.dtype):
                print(f"\n\n\ndtype mismatch for {proj_bias} (original: {state_dict[proj_bias].dtype}, new: {self.proj_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[proj_bias] = self.proj_bias

        if layer_norm_2_weight:
            # check if the shapes and types are correct
            if(state_dict[layer_norm_2_weight].shape != self.layer_norm_2_weight.shape):
                print(f"\n\n\nshape mismatch for {layer_norm_2_weight} (original: {state_dict[layer_norm_2_weight].shape}, new: {self.layer_norm_2_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[layer_norm_2_weight].dtype != self.layer_norm_2_weight.dtype):
                print(f"\n\n\ndtype mismatch for {layer_norm_2_weight} (original: {state_dict[layer_norm_2_weight].dtype}, new: {self.layer_norm_2_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[layer_norm_2_weight] = self.layer_norm_2_weight

        if layer_norm_2_bias:
            # check if the shapes and types are correct
            if(state_dict[layer_norm_2_bias].shape != self.layer_norm_2_bias.shape):
                print(f"\n\n\nshape mismatch for {layer_norm_2_bias} (original: {state_dict[layer_norm_2_bias].shape}, new: {self.layer_norm_2_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[layer_norm_2_bias].dtype != self.layer_norm_2_bias.dtype):
                print(f"\n\n\ndtype mismatch for {layer_norm_2_bias} (original: {state_dict[layer_norm_2_bias].dtype}, new: {self.layer_norm_2_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[layer_norm_2_bias] = self.layer_norm_2_bias

        if mlp_layer1_weight:
            # check if the shapes and types are correct
            if(state_dict[mlp_layer1_weight].shape != self.mlp_layer1_weight.shape):
                print(f"\n\n\nshape mismatch for {mlp_layer1_weight} (original: {state_dict[mlp_layer1_weight].shape}, new: {self.mlp_layer1_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[mlp_layer1_weight].dtype != self.mlp_layer1_weight.dtype):
                print(f"\n\n\ndtype mismatch for {mlp_layer1_weight} (original: {state_dict[mlp_layer1_weight].dtype}, new: {self.mlp_layer1_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[mlp_layer1_weight] = self.mlp_layer1_weight

        if mlp_layer1_bias:
            # check if the shapes and types are correct
            if(state_dict[mlp_layer1_bias].shape != self.mlp_layer1_bias.shape):
                print(f"\n\n\nshape mismatch for {mlp_layer1_bias} (original: {state_dict[mlp_layer1_bias].shape}, new: {self.mlp_layer1_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[mlp_layer1_bias].dtype != self.mlp_layer1_bias.dtype):
                print(f"\n\n\ndtype mismatch for {mlp_layer1_bias} (original: {state_dict[mlp_layer1_bias].dtype}, new: {self.mlp_layer1_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[mlp_layer1_bias] = self.mlp_layer1_bias

        if mlp_layer2_weight:
            # check if the shapes and types are correct
            if(state_dict[mlp_layer2_weight].shape != self.mlp_layer2_weight.shape):
                print(f"\n\n\nshape mismatch for {mlp_layer2_weight} (original: {state_dict[mlp_layer2_weight].shape}, new: {self.mlp_layer2_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[mlp_layer2_weight].dtype != self.mlp_layer2_weight.dtype):
                print(f"\n\n\ndtype mismatch for {mlp_layer2_weight} (original: {state_dict[mlp_layer2_weight].dtype}, new: {self.mlp_layer2_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[mlp_layer2_weight] = self.mlp_layer2_weight

        if mlp_layer2_bias:
            # check if the shapes and types are correct
            if(state_dict[mlp_layer2_bias].shape != self.mlp_layer2_bias.shape):
                print(f"\n\n\nshape mismatch for {mlp_layer2_bias} (original: {state_dict[mlp_layer2_bias].shape}, new: {self.mlp_layer2_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[mlp_layer2_bias].dtype != self.mlp_layer2_bias.dtype):
                print(f"\n\n\ndtype mismatch for {mlp_layer2_bias} (original: {state_dict[mlp_layer2_bias].dtype}, new: {self.mlp_layer2_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[mlp_layer2_bias] = self.mlp_layer2_bias
        
        if v_scale_weight:
            # check if the shapes and types are correct
            if(state_dict[v_scale_weight].shape != self.v_weight.shape):
                print(f"\n\n\nshape mismatch for {v_scale_weight} (original: {state_dict[v_scale_weight].shape}, new: {self.v_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_scale_weight].dtype != self.v_weight.dtype):
                print(f"\n\n\ndtype mismatch for {v_scale_weight} (original: {state_dict[v_scale_weight].dtype}, new: {self.v_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[v_scale_weight] = self.v_weight

        if v_shift_weight:
            # check if the shapes and types are correct
            if(state_dict[v_shift_weight].shape != self.v_weight.shape):
                print(f"\n\n\nshape mismatch for {v_shift_weight} (original: {state_dict[v_shift_weight].shape}, new: {self.v_weight.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_shift_weight].dtype != self.v_weight.dtype):
                print(f"\n\n\ndtype mismatch for {v_shift_weight} (original: {state_dict[v_shift_weight].dtype}, new: {self.v_weight.dtype})\n\n\n")
                raise ValueError

            state_dict[v_shift_weight] = self.v_weight
        
        if v_scale_bias:
            # check if the shapes and types are correct
            if(state_dict[v_scale_bias].shape != self.v_bias.shape):
                print(f"\n\n\nshape mismatch for {v_scale_bias} (original: {state_dict[v_scale_bias].shape}, new: {self.v_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_scale_bias].dtype != self.v_bias.dtype):
                print(f"\n\n\ndtype mismatch for {v_scale_bias} (original: {state_dict[v_scale_bias].dtype}, new: {self.v_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[v_scale_bias] = self.v_bias

        if v_shift_bias:
            # check if the shapes and types are correct
            if(state_dict[v_shift_bias].shape != self.v_bias.shape):
                print(f"\n\n\nshape mismatch for {v_shift_bias} (original: {state_dict[v_shift_bias].shape}, new: {self.v_bias.shape})\n\n\n")
                raise ValueError
            if(state_dict[v_shift_bias].dtype != self.v_bias.dtype):
                print(f"\n\n\ndtype mismatch for {v_shift_bias} (original: {state_dict[v_shift_bias].dtype}, new: {self.v_bias.dtype})\n\n\n")
                raise ValueError

            state_dict[v_shift_bias] = self.v_bias
            
        return state_dict




def get_pretained_weight_loaded_style_transformer_state_dict(
        state_dict: OrderedDict,
        shifted_window_block_path,
        encoder_dim,
        decoder_dim,
        encoder_mlp_ratio,
        decoder_mlp_ratio,
        encoder_window_size,
        decoder_window_size,
        encoder_qkv_bias=True,
        decoder_qkv_bias=True,
        encoder_proj_bias=True,
        decoder_proj_bias=True,
        encoder_norm_layer=None,
        decoder_norm_layer=nn.LayerNorm,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
        decoder_exclude_MLP_after_Fcs_self_MHA=False,
    ):
    """
    Load the pre-trained weights to the style transformer
    """

    # creat the weight loader
    pretrained_weight_loader = PretrainedSwinWeightsLoader(shifted_window_block_path=shifted_window_block_path)

    # load the weights
    state_dict = pretrained_weight_loader.load_weights_to_style_transformer(
        state_dict,
        encoder_dim=encoder_dim,
        decoder_dim=decoder_dim,
        encoder_mlp_ratio=encoder_mlp_ratio,
        decoder_mlp_ratio=decoder_mlp_ratio,
        encoder_window_size=encoder_window_size,
        decoder_window_size=decoder_window_size,
        encoder_qkv_bias=encoder_qkv_bias,
        decoder_qkv_bias=decoder_qkv_bias,
        encoder_proj_bias=encoder_proj_bias,
        decoder_proj_bias=decoder_proj_bias,
        encoder_norm_layer=encoder_norm_layer,
        decoder_norm_layer=decoder_norm_layer,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end=decoder_use_regular_MHA_instead_of_Swin_at_the_end,
        decoder_exclude_MLP_after_Fcs_self_MHA=decoder_exclude_MLP_after_Fcs_self_MHA,
    )

    # return the loaded state dict
    return state_dict


    
    


if __name__ == "__main__":

    import sys
    import os
    import torch.nn as nn

    # add the project path to the sys path
    project_absolute_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    sys.path.append(project_absolute_path)

    from codes.style_transformer import StyleTransformer


    # create the model
    style_transformer = StyleTransformer(
        encoder_dim=256,
        decoder_dim=256,
        encoder_num_heads=8,
        decoder_num_heads=8,
        encoder_window_size=[7, 7],
        decoder_window_size=[7, 7],
        encoder_shift_size=[4, 4],
        decoder_shift_size=[4, 4],
        encoder_mlp_ratio=4.0,
        decoder_mlp_ratio=4.0,
        encoder_dropout=0.0,
        decoder_dropout=0.0,
        encoder_attention_dropout=0.0,
        decoder_attention_dropout=0.0,
        encoder_qkv_bias=True,
        decoder_qkv_bias=True,
        encoder_proj_bias=True,
        decoder_proj_bias=True,
        encoder_stochastic_depth_prob=0.1,
        decoder_stochastic_depth_prob=0.1,
        encoder_norm_layer=None,
        decoder_norm_layer=nn.LayerNorm,
        encoder_MLP_activation_layer=nn.GELU,
        decoder_MLP_activation_layer=nn.GELU,
        encoder_if_use_processed_Key_in_Scale_and_Shift_calculation=True,
        decoder_use_instance_norm_with_affine=False,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
        decoder_use_Key_instance_norm_after_linear_transformation=True,
        decoder_exclude_MLP_after_Fcs_self_MHA=False
    )
    

    
    # get the state dict of the model
    state_dict = style_transformer.state_dict()

    # get the save of the state dict
    state_dict_save = state_dict.copy()

    # get the key list
    key_list = list(state_dict.keys())


    # load the weights
    state_dict = get_pretained_weight_loaded_style_transformer_state_dict(
        state_dict = state_dict,
        shifted_window_block_path = "weights/model_basic_layer_1_module_list_shifted_window_block_state_dict.pth",
        encoder_dim=256,
        decoder_dim=256,
        encoder_mlp_ratio=4,
        decoder_mlp_ratio=4,
        encoder_window_size=[7, 7],
        decoder_window_size=[7, 7],
        encoder_qkv_bias=True,
        decoder_qkv_bias=True,
        encoder_proj_bias=True,
        decoder_proj_bias=True,
        encoder_norm_layer=None,
        decoder_norm_layer=nn.LayerNorm,
        decoder_use_regular_MHA_instead_of_Swin_at_the_end=False,
        decoder_exclude_MLP_after_Fcs_self_MHA=False,
    )


    for key in key_list:
        if torch.all(state_dict_save[key] == state_dict[key]):
            print(f"\n\n{key} IS NOT CHANGED IN PRETRAINED WEIGHT LOADING!!!!!!\n\n")


