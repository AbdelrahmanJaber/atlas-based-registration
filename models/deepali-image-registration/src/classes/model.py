import torch
import deepali.networks.layers as layers
from deepali.core.grid import Grid

from deepali.spatial.transformer import ImageTransformer
from deepali.spatial.nonrigid import DisplacementFieldTransform

import torch.nn.functional as F

class DeepAliUNet(torch.nn.Module):
    '''
    DeepAli UNet model
    '''
    def __init__(self, input_channels=2, out_channels=2, features=32):
        '''
        Constructor for the DeepAliUNet class

        :param input_channels: Number of input channels
        :param out_channels: Number of output channels
        :param features: Number of features

        :return: None
        '''
        super(DeepAliUNet, self).__init__()
        self.flow_scaling_factor = 1

        # Encoder
        self.encoder_conv1 = layers.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.encoder_conv2 = layers.Conv2d(16, features, kernel_size=3, stride=2, padding=1)
        self.encoder_conv3 = layers.Conv2d(features, features, kernel_size=3, stride=2, padding=1)
        self.encoder_conv4 = layers.Conv2d(features, features, kernel_size=3, stride=2, padding=1)
        self.encoder_conv5 = layers.Conv2d(features, features, kernel_size=3, stride=2, padding=1)

        # Decoder
        self.decoder_conv4 = layers.Conv2d(features, features, kernel_size=3, padding=1)
        self.decoder_conv3 = layers.Conv2d(2 * features, features, kernel_size=3, padding=1)
        self.decoder_conv2 = layers.Conv2d(2 * features, features, kernel_size=3, padding=1)
        self.decoder_conv1 = layers.Conv2d(2 * features, 16, kernel_size=3, padding=1)

        # Final convolutions
        self.final_conv1 = layers.Conv2d(32, 16, kernel_size=3, padding=1)
        self.final_conv2 = layers.Conv2d(16, out_channels, kernel_size=3, padding=1)

        self.activation = layers.Activation("leakyrelu", 0.2) # Leaky ReLU activation function

    def forward(self, x):
        '''
        Forward pass for the DeepAliUNet model

        :param x: Input tensor

        :return: Flow tensor
        '''
        # Encoder
        enc1 = self.activation(self.encoder_conv1(x)) 
        enc2 = self.activation(self.encoder_conv2(enc1))  
        enc3 = self.activation(self.encoder_conv3(enc2))  
        enc4 = self.activation(self.encoder_conv4(enc3)) 
        enc5 = self.activation(self.encoder_conv5(enc4)) 
        
        # Decoder
        # Upsample and concatenate
        dec4 = F.interpolate(enc5, scale_factor=2, mode='bilinear', align_corners=True)  
        dec4 = self.activation(self.decoder_conv4(dec4))  
        dec4 = torch.cat((dec4, enc4), dim=1) 

        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True) 
        dec3 = self.activation(self.decoder_conv3(dec3)) 
        dec3 = torch.cat((dec3, enc3), dim=1)  

        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)  
        dec2 = self.activation(self.decoder_conv2(dec2))  
        dec2 = torch.cat((dec2, enc2), dim=1)  

        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)  
        dec1 = self.activation(self.decoder_conv1(dec1)) 
        dec1 = torch.cat((dec1, enc1), dim=1)  

        # Final convolutions
        final1 = self.activation(self.final_conv1(dec1))  
        flow = self.final_conv2(final1)  

        flow = flow * self.flow_scaling_factor # Scale the flow

        return flow

class DeepaliSpatialTransformer(torch.nn.Module):
    '''
    DeepAli spatial transformer
    '''
    def __init__(self, grid):
        '''
        Constructor for the DeepaliSpatialTransformer class

        :param grid: Grid object

        :return: None
        '''
        super(DeepaliSpatialTransformer, self).__init__()
        self.grid = grid

    def forward(self, input, flow):
        '''
        Forward pass for the DeepaliSpatialTransformer model

        :param input: Input tensor
        :param flow: Flow tensor

        :return: Output tensor
        '''
        # Create the transformer
        transformer= ImageTransformer(
            transform=DisplacementFieldTransform(self.grid, groups=flow.shape[0], params=flow, resize=False),
            align_centers= True
        )
        output = transformer(input)
        return output

class VoxelMorph(torch.nn.Module):
    '''
    VoxelMorph model
    '''
    def __init__(self, grid_size, auxiliary_data=False):
        '''
        Constructor for the VoxelMorph class

        :param grid_size: Grid size

        :return: None
        '''
        super(VoxelMorph, self).__init__()
        self.auxiliary_data = auxiliary_data
        self.Unet = DeepAliUNet() # Create the DeepAli UNet model
        self.spatial_transformer = DeepaliSpatialTransformer(grid=Grid(size=grid_size, align_corners=True)) # Create the DeepAli spatial transformer
    
    def forward(self, source, target):
        '''
        Forward pass for the VoxelMorph model
        '''
        if self.auxiliary_data:
            return self._forward_with_auxiliary(source, target)
        
        return self._forward_without_auxiliary(source, target)

    def _forward_with_auxiliary(self, source, target):
        '''
        Forward pass for the VoxelMorph model with auxiliary data
        '''
        source, source_seg = source[0], source[1] # Get the source and source segmentation
        x = torch.cat([source, target], dim=1) # Concatenate the source and target
        flow = self.Unet(x) 
        output = self.spatial_transformer(source, flow) 
        output_seg = self.spatial_transformer(source_seg, flow) # Transform the source segmentation
        return output, flow, output_seg

    def _forward_without_auxiliary(self, source, target):
        '''
        Forward pass for the VoxelMorph model without auxiliary data
        '''
        x = torch.cat([source, target], dim=1) # Concatenate the source and target
        flow = self.Unet(x)
        output = self.spatial_transformer(source, flow)
        return output, flow

