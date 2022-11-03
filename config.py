import numpy as np
import wget
import torch
import os
import urllib

class Base:
    img_size : int = 224
    patch_size : int = 16
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
    
    if (pretrained == True or fine_tuned == True):
        if config_name == "B_16":
            if pretrained is True and fine_tuned is False:
                url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz"

            elif fine_tuned and pretrained:
                url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_16.npz"
                base.out_channels = 1000
                base.img_size = 384

        elif config_name == "B_32":
            base.patch_size = 32
            if pretrained is True and fine_tuned is False:
                url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz"

            elif fine_tuned and pretrained:
                url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-B_32.npz"
                base.out_channels = 1000
                base.img_size = 384


        elif config_name == "L_16":
            base.layers = 24
            base.d_size = 1024
            base.mlp_size = 4096
            base.n_heads = 16

            if pretrained is True and fine_tuned is False:
                url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz"

            elif fine_tuned and pretrained:
                url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_16.npz"
                base.out_channels = 1000
                base.img_size = 384
        
        elif config_name == "L_32":
            base.layers = 24
            base.d_size = 1024
            base.mlp_size = 4096
            base.n_heads = 16
            base.patch_size = 32

            if pretrained is True and fine_tuned is False:
                url = "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz"

            elif fine_tuned and pretrained:
                url = "https://storage.googleapis.com/vit_models/imagenet21k%2Bimagenet2012/ViT-L_32.npz"
                base.out_channels = 1000
                base.img_size = 384

        url = urllib.parse.unquote(url)
        print(url)
    return base, url 

def get_weights(config_name, pretrained, fine_tuned):
    config, url = get_config(config_name, pretrained, fine_tuned)

    if url != "":
        filename = wget.download(url, out="")
        npy_weights = np.load(filename, allow_pickle=True) # numpy weights
        os.remove(filename)
        pre_logits, torch_weights = rename_weights(npy_weights) # convert numpy to torch weights and rename them

    else:
        pre_logits = False
        torch_weights = 0

    return config, pre_logits, torch_weights

def load_weights(torch_weights, model):
    matches = []
    # t_n -> torch name, t_w -> torch weight, m_n -> model name, m_w -> model weight
    for (t_n, t_w), (m_n, m_w) in zip(sorted(torch_weights.items()), sorted(model.state_dict().items())):
        # Checking if shapes and names are correct

        if t_n == m_n and t_w.shape == m_w.shape:
            matches.append(True)
        else:
            matches.append(False)
            print(t_n, " : ", t_w.shape, " -> ", m_n, " : ", m_w.shape)
            print("Don't match.")


    # Checking if all shapes are correct
    assert all(matches), " Some of the weights are different than in the original model. "
    
    model.load_state_dict(torch_weights)
    check(model.state_dict(), torch_weights)
    model.eval()
    return model

def check(model_dict, torch_weights):
    matches = []
    for (t_n, t_w), (m_n, m_w) in zip(sorted(torch_weights.items()), sorted(model_dict.items())):
        # Checking if weights are the same 
        matches.append(True) if torch.equal(t_w, m_w) else matches.append(False)
        
    # Checking if all shapes are correct
    assert all(matches), "Error : Weights are not assigned properly. " 
    print("All weights are correct!")

def rename_weights(npy_weights):
    torch_weights = {}
    fixed_w = {}
    pre_logits = False

    for name, weight in npy_weights.items():
        n = name.replace("/", ".")
        w = torch.from_numpy(weight)
        
        n = n.replace("Transformer", "")
        n = n.replace("encoderblock_", "vit_blocks.")

        n = n.replace("LayerNorm_0", "ln1")
        n = n.replace("LayerNorm_2", "ln2")

        n = n.replace("MlpBlock_3.Dense_0", "mlp.fc1")
        n = n.replace("MlpBlock_3.Dense_1", "mlp.fc2")

        n = n.replace("MultiHeadDotProductAttention_1", "mha")

        n = n.replace("embedding", "embeddings.projection")
        n = n.replace("posembed_input.pos_embeddings.projection", "embeddings.positions")
        n = n.replace("cls", "embeddings.cls_token")
        
        n = n.replace("kernel", "weight")
        n = n.replace("scale", "weight")
        n = n.replace("out", "linear") 

        n = n.replace("pre_logits", "pre_logits_layer")

        if "query" in n:
            n = n.replace("query", "Q")        
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "key" in n:
            n = n.replace("key", "K")
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "value" in n:
            n = n.replace("value", "V")
            if "weight" in n :
                w = w.reshape(w.shape[0], w.shape[1] * w.shape[2])
            if "bias" in n:
                w = w.reshape(-1)

        elif "linear" in n:
            if "weight" in n:
                w = w.reshape(w.shape[0] * w.shape[1], w.shape[2])

        elif "fc1" in n or "fc2" in n:
            if "weight" in n:
                w = w.reshape(w.shape[1], w.shape[0])

        elif "embeddings.positions" in n:
            w = w.reshape(w.shape[0], w.shape[2], w.shape[1])

        elif "embeddings.cls_token" in n:
            w = w.reshape(w.shape[0], w.shape[2], w.shape[1])

        elif "head" in n:
            if "weight" in n:
                w = w.reshape(w.shape[1], w.shape[0])

        elif "embeddings.projection" in n:
            if "weight" in n:
                p, _, c, d = w.shape
                w = w.reshape(d, c, p, p)

        if n[0] == ".":
            n = n[1:]

        if "pre_logits" in n:
            pre_logits = True

        torch_weights[n] = w

        print(name, " -> ", n)


    return pre_logits, torch_weights

def get_labels(pretrained=False, fine_tuned=False):
    labels = {}
    if fine_tuned and pretrained:
        f = open("Vision-Transformer/labels/ImageNet1k.txt", "r")
        for i in f:
            line = i.split(":")
            # labels[class] = name
            labels[int(line[0].replace("{", ""))] = line[1][1:-2].replace("'", "")
            
        f.close()

    if pretrained and fine_tuned==False:
        f = open("Vision-Transformer/labels/ImageNet21k.txt", "r")
        for i, k in enumerate(f):
            # labels[class] = name
            labels[i] = k.replace("\n", "")

        f.close()

    return labels
