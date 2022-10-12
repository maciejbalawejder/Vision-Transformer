class Base:
    img_size : int = 244
    n_patches : int = 16
    layers : int = 12
    d_size : int = 768
    mlp_size : int = 3072
    n_heads : int = 12
    p_att : float = 0.1
    p_emb : float = 0.1
    p_mlp : float = 0.1
    out_channels : int = 21843
    eps : float = 1e-6
    pre_logits = True

def get_config(config_name, pretrained, fine_tuned):
    url = ""
    base = Base()
    if config_name == "B_16":
        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"

        elif fine_tuned:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz"
            base.classes = 1000

    elif config_name == "B_32":
        base.n_patches = 32
        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz"

        elif fine_tuned:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz"
            base.classes = 1000

    elif config_name == "L_16":
        base.layers = 24
        base.d_size = 1024
        base.mlp_size = 4096
        base.heads = 16

        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz"

        elif fine_tuned:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz"
            base.classes = 1000
    
    elif config_name == "L_32":
        base.layers = 24
        base.d_size = 1024
        base.mlp_size = 4096
        base.heads = 16
        base.n_patches = 32

        if pretrained:
            url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"

        elif fine_tuned:
            url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz"
            base.classes = 1000

    return base, url 


    




