# @Time : 2024/3/22 10:33
# @Author : Zeyong Ji
import copy
from torch import nn
from torch.nn import  ModuleList
from models.module.Block import MultiHeadSelfAttention,MultiHeadCrossAttention,FFN,PositionalEncoding

'''
自己编写的Transformer
'''

# 多模态Transformer
class Transformer(nn.Module):
    def __init__(self,dim,num_layers=6,num_heads=8,qkv_bias=False, qk_scale=None,dropout=0.1):
        super().__init__()
        decoder_layer = TranfsformerDecoderlayer(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.decoder = TranfsformerDecoder(decoder=decoder_layer,num_layers=num_layers)
        encoder_layer = TranfsformerEncoderLayer(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.encoder = TranfsformerEncoder(dim=dim,encoder=encoder_layer,num_layers=num_layers)

    def forward(self,src,tgt):
        src = self.encoder(src)
        output = self.decoder(src,tgt)
        return output

# 解码器
class TranfsformerDecoder(nn.Module):
    def __init__(self,decoder = None,num_layers=6):
        super().__init__()
        assert decoder != None,"Encoder cannot be None"
        self.decoder_layers = self.get_clones(decoder,num_layers)

    def forward(self,src,tgt):
        output = tgt
        for model in self.decoder_layers:
            output = model(src,output)
        return output

    def get_clones(self, module, num_layers):
        return ModuleList([copy.deepcopy(module) for _ in range(num_layers)])


# 解码器层
class TranfsformerDecoderlayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.cross_attn = MultiHeadCrossAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.ffn = FFN(dim)
    def forward(self,src,tgt):
        # 递进式，并行式
        tgt = tgt + self.self_attn(tgt)
        tgt = tgt + self.cross_attn(src,tgt)
        tgt = self.ffn(tgt)
        return tgt

# 编码器
class TranfsformerEncoder(nn.Module):
    def __init__(self,dim,encoder=None,num_layers=6):
        super().__init__()
        assert encoder != None,"Encoder cannot be None"
        self.pos_encod = PositionalEncoding(d_model=dim)
        self.encoder_layers = self.get_clones(encoder,num_layers)

    def get_clones(self,module,num_layers):
        return ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

    def forward(self,src):
        output = src
        for model in self.encoder_layers:
            output = model(src)
        return output

# 编码器层
class TranfsformerEncoderLayer(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.ffn = FFN(dim)

    def forward(self,src):
        src = src + self.self_attn(src)
        src = self.ffn(src)
        return src




