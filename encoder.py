import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math, copy
import numpy as np
import time
from torch.autograd import Variable
import torchvision.models as models

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=7000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        '''这个地方去掉，做pe的对照实验'''
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, compress_attention=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.compress_attention = compress_attention
        self.compress_attention_linear = nn.Linear(h, 1)
        self.boxes_vector_linear = nn.Linear(512, 256)

    def forward(self, query, key, value, boxes_vector=None, mask=None, align=None):
        "Implements Figure 2"
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # print("在Multi中，query的尺寸为", query.shape)
        # print("boxes_vector",boxes_vector.shape)

        B, H, N, D = query.shape

        # 遮挡实验
        if boxes_vector != None:
            boxes_vector = self.boxes_vector_linear(boxes_vector)
            query = query.unsqueeze(2).repeat(1, 1, N, 1, 1)
            key = key.unsqueeze(2).repeat(1, 1, N, 1, 1)
            value = value.unsqueeze(2).repeat(1, 1, N, 1, 1)

            for b in range(B):
                for n in range(N):
                    query[b, :, n, n, :] = boxes_vector[b, n, :]
                    key[b, :, n, n, :] = boxes_vector[b, n, :]
                    value[b, :, n, n, :] = boxes_vector[b, n, :]

            query = query.contiguous().view(B, H, N, N * D)
            key = key.contiguous().view(B, H, N, N * D)
            value = value.contiguous().view(B, H, N, N * D)

        # 2) Apply attention on all the projected vectors in batch.
        x, attention_map = attention(query, key, value, mask=mask, dropout=self.dropout, align=align)

        # 3) "Concat" using a view and apply a final linear.
        if boxes_vector != None:
            x = x.transpose(1, 2).contiguous().view(nbatches, N, N, self.h * self.d_k)
        else:
            x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        # 因为multi-head可能会产生若干张注意力map，应该将其合并为一张
        if self.compress_attention:
            batch, head, s1, s2 = attention_map.shape
            attention_map = attention_map.permute(0, 2, 3, 1).contiguous()
            attention_map = self.compress_attention_linear(attention_map).permute(0, 3, 1, 2).contiguous()
        # print('transformer file:', attention_map)

        return self.linears[-1](x), attention_map

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    '''
    这里使用偏移k=2是因为前面补位embedding
    '''
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None, align=None):
    "Compute 'Scaled Dot Product Attention'"

    # print(mask)
    # print("在attention模块,q_{0}".format(query.shape))
    # print("在attention模块,k_{0}".format(key.shape))
    # print("在attention模块,v_{0}".format(key.shape))
    # print("mask :",mask)
    # print("mask的尺寸为",mask.shape)

    d_k = query.size(-1)

    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    # print(mask.shape,scores.shape)
    if mask is not None:
        # print("scores",scores.shape)
        # print(mask)
        mask = mask.unsqueeze(1)
        # print("mask", mask.shape)
        scores = scores.masked_fill(mask == 0, float('-9999'))
    else:
        pass
        # print("scores", scores.shape)
    '''
    工程
    这里的scores需要再乘上一个prob
    这个prob的序号要和word_index对应！
    '''

    # if align is not None:
    #     # print("score", scores.shape)
    #     # print("align", align.shape)
    #
    #     scores = scores * align.unsqueeze(1)
    # print("scores",scores.shape)
    p_attn = F.softmax(scores, dim=-1)
    # print(p_attn)
    # if mask is not None:
    #     print("p_attn",p_attn)
    if dropout is not None:
        p_attn = dropout(p_attn)
    # print("p_attn", p_attn)
    return torch.matmul(p_attn, value), p_attn

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        # print(features)
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe

class FeatureEnhancer(nn.Module):

    def __init__(self, channel):
        super(FeatureEnhancer, self).__init__()
        self.channel = channel
        self.multihead = MultiHeadedAttention(h=4, d_model=self.channel * 2, dropout=0.1)
        self.mul_layernorm1 = LayerNorm(features=self.channel * 2)

        self.pff = PositionwiseFeedForward(self.channel * 2, self.channel * 2)
        self.mul_layernorm3 = LayerNorm(features=self.channel * 2)

        self.linear = nn.Linear(self.channel * 2, self.channel)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, conv_feature):
        '''
        text : (batch, seq_len, embedding_size)
        global_info: (batch, embedding_size, 1, 1)
        conv_feature: (batch, channel, H, W)
        '''
        batch, cc, h, w = conv_feature.shape
        conv_feature = conv_feature.view(batch, conv_feature.shape[1], -1)

        position2d = positionalencoding2d(self.channel, h, w).float().cuda().unsqueeze(0).view(1, self.channel, h * w)
        position2d = position2d.repeat(batch, 1, 1)
        # print("conv_feature",conv_feature.shape,position2d.shape)
        conv_feature = torch.cat([conv_feature, position2d], 1)  # batch, 128(64+64), 32, 128
        # print("conv_feature", conv_feature.shape)
        result = conv_feature.permute(0, 2, 1).contiguous()
        origin_result = result
        result = self.mul_layernorm1(origin_result + self.multihead(result, result, result, mask=None)[0])
        origin_result = result
        result = self.mul_layernorm3(origin_result + self.pff(result))
        result = self.linear(result).permute(0, 2, 1).contiguous().view(batch, cc, h, w)
        result = self.avgpool(result).view(result.size(0), -1)
        return result

