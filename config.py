class Config:
    img_size : int = 144
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    patch_size : int = 16
    att_drop : float = 0.1
    pos_drop : float = 0.1
    mlp_drop : float = 0.1