import torch
import torch.nn as nn
from torch import Tensor
import math
from config import Config


class Embedding(nn.Module):
    """ Patch and Position Embeddings + CLS Token. 

    Parameters
    ----------
    in_channels : int - input channels of the image
    patch_size : int - size of the patch
    img_size : int - height or width of the image assuming that it's square
    d_size : int - embedding dimension from config class
    p : float - dropout rate

    Attributes
    ----------
    projection : nn.Conv2d - projection of input image to the embedding patches 
    n_patches : int - number of patches
    positions : nn.Parameter - position embeddings randomly intialized at the start
    cls_token : nn.Parameter - learnable classification token(totally unnecessary)
    pos_drop : nn.Dropout - embedding dropout
    
    """

    def __init__(
            self,
            in_channels,
            patch_size,
            img_size,
            d_size,
            p
            ):

            super().__init__()
            
            self.projection = nn.Conv2d(
                in_channels = in_channels,
                out_channels = d_size,
                kernel_size = patch_size,
                stride = patch_size
            )
            
            self.n_patches = img_size // patch_size
            self.positions = nn.Parameter(data = torch.zeros(size=(1, d_size, self.n_patches + 1)))
            self.cls_token = nn.Parameter(data = torch.zeros(size=(1, d_size, 1)))
            self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Returns
        -------
        Tensor - with shape (batch, d_size, n_patches + 1)

        """

        batch, in_channels, height, width = x.shape
        patch_emb = self.projection(x).flatten(2) # shape : (batch, d_size, n_patches)
        patch_emb = torch.cat((self.cls_token.expand(batch, -1, -1), patch_emb), axis=-1) # shape : (batch, d_size, n_patches + 1)
        pos_emb = patch_emb + self.positions
        return self.dropout(pos_emb)






        




