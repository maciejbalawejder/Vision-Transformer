import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
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
            
            self.n_patches = (img_size // patch_size)**2
            self.positions = nn.Parameter(data = torch.zeros(size=(1, d_size, self.n_patches + 1)))
            self.cls_token = nn.Parameter(data = torch.zeros(size=(1, d_size, 1)))
            self.dropout = nn.Dropout(p=p)
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, in_channels, height, width)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + 1, d_size)

        """

        batch, in_channels, height, width = x.shape
        patch_emb = self.projection(x).flatten(2) # shape : (batch, d_size, n_patches)
        patch_emb = torch.cat((self.cls_token.expand(batch, -1, -1), patch_emb), axis=-1) # shape : (batch, d_size, n_patches + 1)
        pos_emb = patch_emb + self.positions
        return self.dropout(pos_emb.permute(0, 2, 1))

class MultiHeadAttention(nn.Module):
    """ Attention mechanism. 

    Parameters
    ----------

    d_size : int - embedding dimension from config class
    n_heads : int - number of heads 
    p : float - dropout rate

    Attributes
    ----------
    c : nn.Linear - qkv projection using single layer
    linear : nn.Linear - final projection in attention mechanism
    att_drop : nn.Dropout - attention dropout layer
    

    """

    def __init__(
        self,
        d_size,
        n_heads,
        p
        ):

        super().__init__()

        self.c = nn.Linear(d_size, d_size * 3)
        self.linear = nn.Linear(d_size, d_size)
        self.att_drop = nn.Dropout(p=p)
        self.n_heads = n_heads
        self.head_dim = d_size // n_heads
        
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)

        """

        batch, n_patches, d_size = x.shape
        mask = torch.triu(input = torch.ones((n_patches, n_patches))).expand(batch, 1, n_patches, n_patches)

        qkv = self.c(x)
        q, k, v = torch.split(
            tensor = qkv, 
            split_size_or_sections = d_size, 
            dim = 2
        ) 

        q = q.reshape(batch, n_patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, n_patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, n_patches, self.n_heads, self.head_dim)

        QK = torch.einsum("bqhd, bkhd -> bhqk", [q, k]) / math.sqrt(d_size)
        QK = QK.masked_fill(mask==0, torch.finfo(torch.float32).min)

        scores = self.att_drop(F.softmax(QK, dim=3))
        output = torch.einsum("bhqk, bvhd -> bqhd", [scores, v])

        concat = output.reshape(batch, n_patches, d_size)
        return self.att_drop(self.linear(concat))
        
class MLP(nn.Module):
    """ Feed Forward module. 

    Parameters
    ----------

    d_size : int - embedding dimension from config class
    mlp_size : int - expansion dimension in mlp module
    p : float - dropout rate

    Attributes
    ----------
    ff : nn.Sequential - all layers in one module

    """

    def __init__(
        self,
        d_size,
        mlp_size,
        p
        ):

        super().__init__()

        self.ff = nn.Sequential(
            nn.Linear(d_size, mlp_size),
            nn.GELU(),
            nn.Linear(mlp_size, d_size),
            nn.Dropout(p)
        )
    
    def forward(self, x):
        """ Forward function.

        Parameters
        ----------
        x : Tensor - input image with shape (batch, n_patches + cls_token, d_size)

        Outputs
        -------
        Tensor - with shape (batch, n_patches + cls_token, d_size)

        """

        return self.ff(x)






if __name__ == "__main__":
    # Sanity checks
    c = Config()
    img = torch.rand(1, 3, c.img_size, c.img_size)
    emb = Embedding(
        in_channels=3,
        patch_size=c.patch_size,
        img_size=c.img_size,
        d_size=c.d_size,
        p=c.pos_drop
    )
    mha = MultiHeadAttention(
        d_size=c.d_size,
        n_heads=c.heads,
        p=c.att_drop
    )
    
    mlp = MLP(
        c.d_size,
        c.mlp_size,
        c.mlp_drop
    )

    img_emb = emb(img)
    img_mha = mha(img_emb)
    print(mlp(img_mha).shape)
        




