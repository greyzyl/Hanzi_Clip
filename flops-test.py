import torch
from thop import profile

from model_m import CLIP

model = CLIP(embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
             vision_patch_size=32, context_length=30, vocab_size=442, transformer_width=512,
             transformer_heads=8, transformer_layers=3)
input = torch.randn(1, 3, 256, 256) #模型输入的形状,batch_size=1
input_1 = torch.zeros(1,30).long()
flops, params = profile(model, inputs=(input, input_1))
print(flops/1e9,params/1e6) #flops单位G，para单位M
