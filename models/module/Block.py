# @Time : 2024/5/3 9:57
# @Author : Zeyong Ji
import torch
import math
from torch import nn,Tensor
import torch.nn.functional as F
# 多头时间自注意力
class MultiHeadTemporalSelfAttention(nn.Module):
    def __init__(self,dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,drop_path=0.1):
        super().__init__()
        assert dim % num_heads == 0,"d_model is not divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.dim ** -0.5
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = qk_scale or self.head_dim ** -0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.temporal_ln = DyT(dim)
        self.temporal_ln = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.fc = nn.Linear(dim,dim)

    def forward(self,x):
        x = self.temporal_ln(x)
        B = x.shape[0]
        q = self.w_q(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.w_k(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,3,1)
        v = self.w_v(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        output = torch.matmul(attn,v).transpose(1,2).reshape(B,-1,self.dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        output = self.drop_path(output)
        output = self.fc(output)
        # return output,attn
        return output


# 多头时间交叉注意力
class MultiHeadTemporalCrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., drop_path=0.1,):
        super().__init__()
        assert dim % num_heads == 0,"d_model is not divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = qk_scale or self.dim ** -0.5
        self.w_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_v = nn.Linear(dim, dim, bias=qkv_bias)
        self.w_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.scale = qk_scale or self.head_dim ** -0.5
        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        # self.temporal_ln1 = DyT(dim)
        # self.temporal_ln2 = DyT(dim)
        self.temporal_ln1 = nn.LayerNorm(dim)
        self.temporal_ln2 = nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path)
        self.fc = nn.Linear(dim, dim)

    def forward(self, src,tgt):
        src = self.temporal_ln(src)
        tgt = self.temporal_ln(tgt)
        B  = tgt.shape[0]
        q = self.w_q(tgt).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.w_k(src).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 3, 1)
        v = self.w_v(src).view(B, -1, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        attn = torch.matmul(q, k) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        output = torch.matmul(attn, v).transpose(1, 2).reshape(B, -1, self.dim)
        output = self.proj(output)
        output = self.proj_drop(output)
        output = self.drop_path(output)
        output = self.fc(output)
        # return output,attn
        return output


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

# 多头自注意力
class MultiHeadSelfAttention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0,"d_model is not divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.ln = nn.LayerNorm(dim)
        # self.ln = DyT(dim)
        self.w_q = nn.Linear(dim,dim,bias=qkv_bias)
        self.w_v = nn.Linear(dim,dim,bias=qkv_bias)
        self.w_k = nn.Linear(dim,dim,bias=qkv_bias)
        self.fc = nn.Linear(dim,dim,bias=qkv_bias)
        self.scale = qk_scale or self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self,x,mask=None):
        x = self.ln(x)
        B = x.shape[0]
        q = self.w_q(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.w_k(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,3,1)
        v = self.w_v(x).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        attn = torch.matmul(q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask==0,-1e10)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn,v).permute(0,2,1,3).contiguous().view(B,-1,self.dim)
        output = self.fc(output)
        # return output,attn
        return output

# 多头交叉注意力
class MultiHeadCrossAttention(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,dropout=0.1):
        super().__init__()
        assert dim % num_heads == 0,"d_model is not divisible by num_heads"
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        # self.ln1 = DyT(dim)
        # self.ln2 = DyT(dim)
        self.w_q = nn.Linear(dim,dim,bias=qkv_bias)
        self.w_v = nn.Linear(dim,dim,bias=qkv_bias)
        self.w_k = nn.Linear(dim,dim,bias=qkv_bias)
        self.fc = nn.Linear(dim,dim,bias=qkv_bias)
        self.scale = qk_scale or self.head_dim ** -0.5
        self.dropout = nn.Dropout(dropout)

    def forward(self,src,tgt,mask=None):
        src = self.ln1(src)
        tgt = self.ln2(tgt)
        B = src.shape[0]
        q = self.w_q(tgt).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        k = self.w_k(src).view(B,-1,self.num_heads,self.head_dim).permute(0,2,3,1)
        v = self.w_v(src).view(B,-1,self.num_heads,self.head_dim).permute(0,2,1,3)
        attn = torch.matmul(q, k) * self.scale
        if mask is not None:
            attn = attn.masked_fill(mask==0,-1e10)
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        output = torch.matmul(attn,v).permute(0,2,1,3).contiguous().view(B,-1,self.dim)
        output = self.fc(output)
        # return output,attn
        return output


# MLP
class FFN(nn.Module):
    def __init__(self,dim,dim_feedforward=2048,dropout=0.1,bias=True):
       super().__init__()
       self.linear1 = nn.Linear(dim,dim_feedforward,bias=bias)
       self.dropout1 = nn.Dropout(dropout)
       self.linear2 = nn.Linear(dim_feedforward,dim,bias=bias)
       self.dropout2 = nn.Dropout(dropout)
       self.activation = nn.ReLU()
       self.nl = nn.LayerNorm(dim)
       # self.nl = DyT(dim)

    def forward(self,x):
        output = self.dropout2(self.linear2(self.dropout1(self.activation(self.linear1(x)))))
        x = self.nl(x + output)
        return x


# 可学习的时序编码
class TemporalEncoding(nn.Module):
    def __init__(self,num_frames,d_model):
        super().__init__()
        self.time_embed = nn.Parameter(torch.zeros(1,num_frames,d_model))

    def forward(self,x):
        x = x + self.time_embed
        return x


class DyT(nn.Module):
    def __init__(self,dim,alpha_init_value=0.5):
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1)*alpha_init_value)
        self.weight = nn.Parameter(torch.ones(dim))
        self.bias = nn.Parameter(torch.zeros(dim))

    def forward(self,x):
        x = torch.tanh(self.alpha*x)
        return x*self.weight+self.bias

# 正余弦位置编码
class PositionalEncoding(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        out = self.dropout(x)
        return out


class GroupWiseLinear(nn.Module):
    # could be changed to:
    # output = torch.einsum('ijk,zjk->ij', x, self.W)
    # or output = torch.einsum('ijk,jk->ij', x, self.W[0])
    def __init__(self, num_class, hidden_dim, bias=True):
        super().__init__()
        self.num_class = num_class
        self.hidden_dim = hidden_dim
        self.bias = bias

        self.W = nn.Parameter(torch.Tensor(1, num_class, hidden_dim))
        if bias:
            self.b = nn.Parameter(torch.Tensor(1, num_class))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W.size(2))
        for i in range(self.num_class):
            self.W[0][i].data.uniform_(-stdv, stdv)
        if self.bias:
            for i in range(self.num_class):
                self.b[0][i].data.uniform_(-stdv, stdv)

    def forward(self, x):
        # x: B,K,d
        x = (self.W * x).sum(-1)
        if self.bias:
            x = x + self.b
        return x



import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionAggregation(nn.Module):
    def __init__(self, D):
        super(AttentionAggregation, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, D),
            nn.Tanh(),
            nn.Linear(D, 1)
        )

    def forward(self, x):
        # 计算输入特征的注意力权重
        attn_scores = self.attention(x).squeeze(-1)  # (B, k)
        attn_weights = torch.softmax(attn_scores, dim=1).unsqueeze(-1)  # (B, k, 1)
        # 对输入特征进行加权平均
        x = torch.sum(x * attn_weights, dim=-1)  # (B, K)
        return x


class LinearAggregation(nn.Module):
    def __init__(self, D,bias=True):
        super(LinearAggregation, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(D, D//2,bias=bias),
            nn.ReLU(),
            nn.Linear(D//2, D,bias=bias)
        )

    def forward(self, x):
        # 计算输入特征的注意力权重
        x = self.attention(x)
        # x = x.max(-1)[0]
        x = x.sum(-1)
        return x