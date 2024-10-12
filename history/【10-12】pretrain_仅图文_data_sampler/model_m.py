from collections import OrderedDict
from typing import Tuple, Union
import math, copy
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from config import config
from resnet import BasicBlock, ResNet

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1):
        super().__init__()

        # all conv layers have stride 1. an avgpool is performed after the second convolution when stride > 1
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU(inplace=True)

        self.avgpool = nn.AvgPool2d(stride) if stride > 1 else nn.Identity()

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu3 = nn.ReLU(inplace=True)

        self.downsample = None
        self.stride = stride

        if stride > 1 or inplanes != planes * Bottleneck.expansion:
            # downsampling layer is prepended with an avgpool, and the subsequent convolution has stride 1
            self.downsample = nn.Sequential(OrderedDict([
                ("-1", nn.AvgPool2d(stride)),
                ("0", nn.Conv2d(inplanes, planes * self.expansion, 1, stride=1, bias=False)),
                ("1", nn.BatchNorm2d(planes * self.expansion))
            ]))

    def forward(self, x: torch.Tensor):
        identity = x

        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.avgpool(out)
        out = self.bn3(self.conv3(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)
        return out


class AttentionPool2d(nn.Module):
    def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
        super().__init__()
        self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

    def forward(self, x):
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
        x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
        x, _ = F.multi_head_attention_forward(
            query=x, key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )

        return x[0]


class ModifiedResNet(nn.Module):
    """
    A ResNet class that is similar to torchvision's but contains the following changes:
    - There are now 3 "stem" convolutions as opposed to 1, with an average pool instead of a max pool.
    - Performs anti-aliasing strided convolutions, where an avgpool is prepended to convolutions with stride > 1
    - The final pooling layer is a QKV attention instead of an average pool
    """

    def __init__(self, layers, output_dim, heads, input_resolution=224, width=64):
        super().__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        # the 3-layer stem
        self.conv1 = nn.Conv2d(3, width // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width // 2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(width // 2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(width)
        self.relu3 = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(2)

        # residual layers
        self._inplanes = width  # this is a *mutable* variable used during construction
        self.layer1 = self._make_layer(width, layers[0])
        self.layer2 = self._make_layer(width * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(width * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(width * 8, layers[3], stride=2)

        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = AttentionPool2d(input_resolution // 32, embed_dim, heads, output_dim)

    def _make_layer(self, planes, blocks, stride=1):
        layers = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            layers.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        def stem(x):
            x = self.relu1(self.bn1(self.conv1(x)))
            x = self.relu2(self.bn2(self.conv2(x)))
            x = self.relu3(self.bn3(self.conv3(x)))
            x = self.avgpool(x)
            return x

        x = x.type(self.conv1.weight.dtype)
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.attnpool(x)

        return x


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None, dropout=None, align=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    else:
        pass

    p_attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linear_0 = nn.Linear(512, 512)
        self.linear_1 = nn.Linear(512, 512)
        self.linear_2 = nn.Linear(512, 512)
        self.linear_3 = nn.Linear(512, 512)
        # self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)

    def forward(self, query, key, value, mask=None, align=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query = self.linear_0(query).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_1(key).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_2(value).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)

        x, attention_map = attention(query, key, value, mask=mask,
                                     dropout=self.dropout, align=align)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)

        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()

        return self.linear_3(x), attention_map

class FusionBlock(nn.Module):
    def __init__(self, attn_mask=None):
        super(FusionBlock, self).__init__()
        self.multihead = MultiHeadedAttention(h=4, d_model=512, dropout=0.1, compress_attention=False)
        self.mul_layernorm2 = LayerNorm(512)
        self.pff = PositionwiseFeedForward(512, 1024)
        self.mul_layernorm3 = LayerNorm(512)

        self.multihead_1 = MultiHeadedAttention(h=4, d_model=512, dropout=0.1, compress_attention=False)
        self.mul_layernorm2_1 = LayerNorm(512)
        self.pff_1 = PositionwiseFeedForward(512, 1024)
        self.mul_layernorm3_1 = LayerNorm(512)

        self.trans = ResidualAttentionBlock(512, 4, attn_mask)

    def forward(self, x):
        cls = x[:, 0, ].unsqueeze(1)
        text_split = x[:, 1:, ].chunk(2, dim=1)
        text = text_split[0]
        text_s = text_split[1]
        word_image_align, _ = self.multihead(text, text_s, text_s, mask=None)
        result = self.mul_layernorm2(text + word_image_align)
        result = self.mul_layernorm3(result + self.pff(result)) # b l c

        word_image_align_1, _ = self.multihead_1(text_s, text, text, mask=None)
        result_1 = self.mul_layernorm2_1(text_s + word_image_align_1)
        result_1 = self.mul_layernorm3_1(result_1 + self.pff_1(result_1)) # b l c

        res = torch.cat([cls, result, result_1], dim=1) # b 2l c
        res = res.permute(1, 0, 2)
        res = self.trans(res)
        res = res.permute(1, 0, 2)

        return res

class FusionLayer(nn.Module):
    def __init__(self, layers, attn_mask=None):
        super().__init__()
        self.layers = layers
        self.resblocks = nn.Sequential(*[FusionBlock(attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)

class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTransformer(nn.Module):
    def __init__(self, input_resolution: int, patch_size: int, width: int, layers: int, heads: int, output_dim: int):
        super().__init__()
        self.input_resolution = input_resolution
        self.output_dim = output_dim
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        self.positional_embedding = nn.Parameter(scale * torch.randn((input_resolution // patch_size) ** 2 + 1, width))
        self.ln_pre = LayerNorm(width)

        self.transformer = Transformer(width, layers, heads)

        self.ln_post = LayerNorm(width)
        self.proj = nn.Parameter(scale * torch.randn(width, output_dim))

    def forward(self, x: torch.Tensor):
        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.positional_embedding.to(x.dtype)
        x = self.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        x = self.ln_post(x[:, 0, :])

        if self.proj is not None:
            x = x @ self.proj

        return x


class CLIP(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 # vision
                 image_resolution: int,
                 vision_layers: Union[Tuple[int, int, int, int], int],
                 vision_width: int,
                 vision_patch_size: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 stroke_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int
                 ):
        super().__init__()

        # self.context_length = config['max_len']
        self.context_length = context_length


        # 自定义的resnet
        # self.visual = ResNet(num_in=3, block=BasicBlock, layers=[3,4,6,3], embed_dim=embed_dim)
        from resnet50 import ResNet_50, Bottleneck
        self.visual = ResNet_50(Bottleneck, [3, 4, 6, 3])
        # self.visual = ResNet_50(Bottleneck, [3, 4, 6, 3])
        # self.visual = ResNet_50(Bottleneck, [3, 8, 36, 3])

        self.transformer = Transformer(
            width=transformer_width,
            layers=3,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )
        self.transformer_s = Transformer(
            width=transformer_width,
            layers=3,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask()
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.stroke_size = stroke_size
        self.token_embedding_s = nn.Embedding(stroke_size, transformer_width)
        self.positional_embedding_s = nn.Parameter(torch.empty(self.context_length, transformer_width))
        self.ln_final_s = LayerNorm(transformer_width)

        self.text_projection_s = nn.Parameter(torch.empty(transformer_width, embed_dim))
        self.logit_scale_s = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        scale = transformer_width ** -0.5
        self.cls = nn.Parameter(scale * torch.randn(transformer_width))
        self.positional_embedding_fl = nn.Parameter(scale * torch.randn( 2 * self.context_length + 1, transformer_width))
        self.ln_pre = LayerNorm(transformer_width)
        # self.fl = FusionLayer(3, attn_mask=self.build_attention_mask_fuse())
        self.fl = FusionLayer(3, attn_mask=None)
        self.ln_post = LayerNorm(transformer_width)
        self.proj = nn.Parameter(scale * torch.randn(transformer_width, 2048))
        self.logit_scale_all = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.logit_scale_i2i = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        nn.init.normal_(self.token_embedding_s.weight, std=0.02)
        nn.init.normal_(self.positional_embedding_s, std=0.01)

        if isinstance(self.visual, ModifiedResNet):
            if self.visual.attnpool is not None:
                std = self.visual.attnpool.c_proj.in_features ** -0.5
                nn.init.normal_(self.visual.attnpool.q_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.k_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.v_proj.weight, std=std)
                nn.init.normal_(self.visual.attnpool.c_proj.weight, std=std)

            for resnet_block in [self.visual.layer1, self.visual.layer2, self.visual.layer3, self.visual.layer4]:
                for name, param in resnet_block.named_parameters():
                    if name.endswith("bn3.weight"):
                        nn.init.zeros_(param)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer_s.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)
            nn.init.normal_(self.text_projection_s, std=self.transformer.width ** -0.5)
            nn.init.normal_(self.proj, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_attention_mask_fuse(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(2 * self.context_length + 1, 2 * self.context_length + 1)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_attention_mask_pos(self):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(15, 15)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    @property
    def dtype(self):
        return self.visual.conv1.weight.dtype

    def encode_image(self, image):
        return self.visual(image.type(self.dtype))

    def encode_text(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        seq_x = x
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection

        return x, seq_x

    def encode_text_s(self, text):
        x = self.token_embedding_s(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding_s.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer_s(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final_s(x).type(self.dtype)

        seq_x = x
        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)] @ self.text_projection_s

        return x, seq_x

    def encode_all(self, text_features, text_features_s):
        all_features = torch.cat([text_features, text_features_s], dim=1)
        all_features_cls = torch.cat([self.cls.to(text_features.dtype) + torch.zeros(text_features.shape[0], 1,
                                                                                     text_features.shape[-1],
                                                                                     dtype=text_features.dtype,
                                                                                     device=text_features.device),
                                      all_features],
                                     dim=1)  # shape = [*, grid ** 2 + 1, width]
        all_features_cls = all_features_cls + self.positional_embedding_fl.to(all_features_cls.dtype)
        # all_features_cls = self.ln_pre(all_features_cls)
        all_features_cls = self.fl(all_features_cls)
        # all_features_cls = self.ln_post(all_features_cls)
        # features_cls = self.ln_post(all_features_cls[:, 0, :])
        features_cls = all_features_cls[:, 0, :]
        features_cls = features_cls @ self.proj
        return features_cls

    def forward(self, image, text, text_s, text_features_fuse=None, test=False):
        image_features = self.encode_image(image)
        if test == False:
            text_features, seq_text = self.encode_text(text)
            text_features_s, seq_text_s = self.encode_text_s(text_s)
            text_features_fuse = self.encode_all(seq_text, seq_text_s)
        else:
            text_features = text
            text_features_s = text_s
            text_features_fuse = text_features_fuse


        # normalized features
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        text_features = text_features / text_features.norm(dim=1, keepdim=True)
        text_features_s = text_features_s / text_features_s.norm(dim=1, keepdim=True)
        text_features_fuse = text_features_fuse / text_features_fuse.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logit_scale_s = self.logit_scale_s.exp()
        logit_scale_all = self.logit_scale_all.exp()
        logit_scale_i2i = self.logit_scale_i2i.exp()
        return image_features, text_features, text_features_s, text_features_fuse, logit_scale, logit_scale_s, logit_scale_all,logit_scale_i2i

