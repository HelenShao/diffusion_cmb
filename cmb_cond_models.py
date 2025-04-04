import torch
from torch import nn
from diffusers import UNet2DModel

class ClassConditionedUnet(nn.Module):
    """
    Original UNet model with 3 layers and attention blocks.
    Uses a UNet2DModel from diffusers with additional conditional input channels.
    """
    def __init__(self, num_cond=1):
        super().__init__()

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information
        self.model = UNet2DModel(
            sample_size=256,           # the target image resolution
            in_channels=1 + num_cond, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            block_out_channels=(32, 64, 64),
            down_block_types=(
                "DownBlock2D",        # a regular ResNet downsampling block
                "AttnDownBlock2D",    # a ResNet downsampling block with spatial self-attention
                "AttnDownBlock2D",
            ),
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",      # a ResNet upsampling block with spatial self-attention
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )

    def forward(self, x, t, condition):
        # Shape of x:
        bs, ch, w, h = x.shape
          
        # x is shape (bs, 1, 256, 256) and condition is (bs, 1, 256, 256)

        # Net input is now x and cond concatenated together along dimension 1
        net_input = torch.cat((x, condition), 1) # (bs, 2, 256, 256)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 256, 256)


class ClassConditionedUnet5Layer(nn.Module):
    """
    Enhanced UNet model with 5 downsampling layers, using only ResNet blocks without attention.
    Uses a UNet2DModel from diffusers with additional conditional input channels.
    """
    def __init__(self, num_cond=1, sample_size=256):
        super().__init__()

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information
        self.model = UNet2DModel(
            sample_size=sample_size,  # Adjust the target image resolution to support 5 downsampling layers
            in_channels=1 + num_cond, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=2,       # how many ResNet layers to use per UNet block
            
            # Five levels of feature maps with increasing channels
            block_out_channels=(32, 64, 128, 256, 512),
            
            # Five downsampling blocks - all using traditional ResNet blocks, no attention
            down_block_types=(
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
            ),
            
            # Five upsampling blocks - all using traditional ResNet blocks, no attention
            up_block_types=(
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )

    def forward(self, x, t, condition):
        # Shape of x:
        bs, ch, w, h = x.shape
          
        # x is shape (bs, 1, 256, 256) and condition is (bs, 1, 256, 256)

        # Net input is now x and cond concatenated together along dimension 1
        net_input = torch.cat((x, condition), 1) # (bs, 2, 256, 256)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 256, 256)


class ClassConditionedUnet5LayerDeep(nn.Module):
    """
    Deep UNet model with 5 layers, more ResNet blocks per layer (3 instead of 2), 
    and larger channel dimensions for more capacity.
    Uses a UNet2DModel from diffusers with additional conditional input channels.
    """
    def __init__(self, num_cond=1, sample_size=256):
        super().__init__()

        # Self.model is an unconditional UNet with extra input channels to accept the conditioning information
        self.model = UNet2DModel(
            sample_size=sample_size,  # Adjust the target image resolution based on your data
            in_channels=1 + num_cond, # Additional input channels for class cond.
            out_channels=1,           # the number of output channels
            layers_per_block=3,       # More ResNet layers per block for additional capacity
            
            # Five levels of feature maps with increasing channels
            block_out_channels=(64, 128, 256, 512, 1024),
            
            # Five downsampling blocks - all using traditional ResNet blocks, no attention
            down_block_types=(
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
                "DownBlock2D",        # a regular ResNet downsampling block
            ),
            
            # Five upsampling blocks - all using traditional ResNet blocks, no attention
            up_block_types=(
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
                "UpBlock2D",          # a regular ResNet upsampling block
              ),
        )

    def forward(self, x, t, condition):
        # Shape of x:
        bs, ch, w, h = x.shape
          
        # x is shape (bs, 1, 256, 256) and condition is (bs, 1, 256, 256)

        # Net input is now x and cond concatenated together along dimension 1
        net_input = torch.cat((x, condition), 1) # (bs, 2, 256, 256)

        # Feed this to the UNet alongside the timestep and return the prediction
        return self.model(net_input, t).sample # (bs, 1, 256, 256)


# Add a dictionary to easily select which model to use
MODEL_REGISTRY = {
    "original": ClassConditionedUnet,
    "unet5layer": ClassConditionedUnet5Layer,
    "unet5layer_deep": ClassConditionedUnet5LayerDeep,
}
