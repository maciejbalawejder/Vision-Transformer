import torch
import torch.nn as nn
import math
from config import Config

class Embedding(nn.Module):
    """ Patch and Position Embeddings. 
    Parameters
    ----------
    in_channels : int - input channels of the image
    patch_size : int - size of the patch
    d_size : int - embedding dimension from config class

    Attributes
    ----------
    projection : nn.Conv2d - projection of input image to the embedding patches 

    
    """

    def __init__(
            self,
            in_channels,
            patch_size,
            d_size
            ):
            
            self.projection = nn.Conv2d(
                in_channels = in_channels,
                out_channels = d_size,
                kernel_size = patch_size,
                stride = patch_size
            )



