class Base:
    img_size : int = 144
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    patch_size : int = 16
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1

class L_16(Base):
    layers : int = 12
    d_size : int = 1024
    mlp_size : int = 4096
    n_heads : int = 16
    patch_size : int = 16
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1

class L_32(Base):
    layers : int = 12
    d_size : int = 1024
    mlp_size : int = 4096
    n_heads : int = 16
    patch_size : int = 32
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1

class B_16(Base):
    img_size : int = 144
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    patch_size : int = 16
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1

class B_32(Base):
    img_size : int = 144
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    patch_size : int = 32
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1


