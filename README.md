# Vision Transformer 
This repo contains PyTorch implementation of ViT model with option to load the pretrained models open-sourced by Google.

### About ViT
ViT is the first succesful applicaton of Transformer architecture to computer vision tasks. It was introducted in 2020 Google's paper called ["An Image is Worth 16x16 Words"](https://arxiv.org/abs/2010.11929). Due to quadratic cost of attention mechanism previous attempts failed to efficient and simple solution. In this paper, they introducted the idea of splitting an images into patches, flatten them and feed them to the transformer encoder. This simple process combined with large dataset achieved the state-of-the-art result on image recogniction task.

If you want to know more about the paper, check out my explanation on [YouTube](https://www.youtube.com/watch?v=D5Ot7VBgPh4&t=5s).

### Installation
First, clone repo and make sure you install all dependencies:

``` 
git clone https://github.com/maciejbalawejder/Vision-Transformer.git
cd Vision-Transformer
pip install requirements.txt
```

If you want to just test the model in the browser, check out [Colab notebook]().

### Usage
```python

img = torch.rand(1, 3, 224, 224)
pretrained, fine_tuned = True, False

vit = ViT("B_16", pretrained, fine_tuned) 
vit.eval()

output = vit(img) # output shape : [1, 1000]

```


#### To-do
- [x] working model
- [x] add different models to config
- [ ] readable readme
- [ ] add pretrain models + colab test notebook
- [ ] add wget library
