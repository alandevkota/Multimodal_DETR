# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding

#******************************added codes*********************************
import torchvision.models as models
from torchvision.models import ResNet50_Weights
#**************************************************************************

#******************************added codes*********************************
'''
Define a New Class for Dual ResNet Backbones:

Create a new class, DualResNetBackbone, which will house two ResNet backbones, one for RGB and the other for IR images.
Initialize both ResNet backbones within this class.
Modify the forward method to process both RGB and IR inputs separately.
Integrate Dual ResNet Backbones into DETR Backbone Structure:

Modify the existing Backbone class to use DualResNetBackbone instead of a single ResNet model.
Ensure that the output of the dual backbones is compatible with the rest of the DETR architecture.
'''

class DualResNetBackbone(nn.Module):
    def __init__(self, backbone_name='resnet50', pretrained=True, return_intermediate_layers=False, trainable_layers=[], dilation=False):
        super(DualResNetBackbone, self).__init__()
        # Dynamically set return_layers based on return_intermediate_layers flag
        if return_intermediate_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
        else:
            # If false, only return the output of the final layer
            return_layers = {'layer4': '0'}
        
        # Load pre-trained ResNet models for both RGB and IR backbones
        self.rgb_backbone = self._load_resnet(backbone_name, pretrained, return_layers)
        self.ir_backbone = self._load_resnet(backbone_name, pretrained, return_layers)

        # Set layers to trainable as specified
        self.set_trainable_layers(trainable_layers)

    def _load_resnet(self, backbone_name, pretrained, return_layers):
        
        if not isinstance(return_layers, dict):
            raise ValueError(f"Expected return_layers to be a dictionary, got {type(return_layers)} instead")
        
        # Load a ResNet model
        model = getattr(models, backbone_name)(pretrained=pretrained)
        # Replace the fully connected layer with an Identity transformation
        model.fc = nn.Identity()

        # Create a new Sequential model that only returns the requested layers
        layers = nn.Sequential()
        for name, module in model.named_children():
            if name in return_layers:
                layers.add_module(name, module)
        return layers

    def set_trainable_layers(self, trainable_layers):
        # Freeze all layers first
        for param in self.rgb_backbone.parameters():
            param.requires_grad = False
        for param in self.ir_backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the specified layers
        for layer_name in trainable_layers:
            for param in getattr(self.rgb_backbone, layer_name).parameters():
                param.requires_grad = True
            for param in getattr(self.ir_backbone, layer_name).parameters():
                param.requires_grad = True

    def forward(self, x_rgb, x_ir):
        features_rgb = self.rgb_backbone(x_rgb)
        features_ir = self.ir_backbone(x_ir)
        return features_rgb, features_ir
#**************************************************************************

class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias



#*****************************modified codes*******************************
# Modify the Backbone class to use DualResNetBackbone
class Backbone(nn.Module):
    """
    Modified Backbone class to incorporate DualResNetBackbone.
    """
    def __init__(self, name: str, train_backbone: bool, num_channels: int, return_intermediate_layers: bool, dilation: bool):
        super().__init__()
        self.body = DualResNetBackbone(name, train_backbone, return_intermediate_layers, dilation)
        self.num_channels = num_channels

    def forward(self, tensor_list_rgb, tensor_list_ir):
        # Directly pass the tensors to the dual backbone without accessing .tensors
        # This assumes tensor_list_rgb and tensor_list_ir are already tensors.
        # If they are NestedTensor instances, you should extract the tensors before this call.
        features_rgb, features_ir = self.body(tensor_list_rgb, tensor_list_ir)
        return features_rgb, features_ir
#**************************************************************************


#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
class Joiner(nn.Module):
    def __init__(self, backbone, position_embedding):
        super(Joiner, self).__init__()
        self.backbone = backbone
        self.position_embedding = position_embedding

    def forward(self, tensor_list_rgb: NestedTensor, tensor_list_ir: NestedTensor):
        # Extract features for both RGB and IR images using the backbone
        features_rgb, features_ir = self.backbone(tensor_list_rgb, tensor_list_ir)
        
        # Assuming position embeddings are computed separately for RGB and IR
        pos_embedding_rgb = self.position_embedding(tensor_list_rgb)
        pos_embedding_ir = self.position_embedding(tensor_list_ir)

        return features_rgb, features_ir, pos_embedding_rgb, pos_embedding_ir

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def build_backbone(args):
    position_embedding = build_position_encoding(args)
    
     # Example: Use a hypothetical args.return_intermediate_layers to control this
    return_intermediate_layers = getattr(args, 'return_intermediate_layers', False)

    # Include return_layers in the call to Backbone
    backbone = Backbone(args.backbone, args.lr_backbone > 0, num_channels=2048, return_layers=return_intermediate_layers, dilation=args.dilation)
    
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

#+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

