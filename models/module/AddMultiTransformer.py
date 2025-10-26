import copy
from torch import nn
from torch.nn import  ModuleList
from models.module.Block import MultiHeadSelfAttention,MultiHeadCrossAttention,MultiHeadTemporalSelfAttention,FFN,TemporalEncoding,PositionalEncoding

# Add + Multi + Transformer
class AddMultiTransformer(nn.Module):
    def __init__(self,dim,num_frames=16,num_layers=6,num_heads=8,qkv_bias=False, qk_scale=None,attn_drop=0.,proj_drop=0.,drop_path=0.1, dropout=0.1):
        super().__init__()
        decoder_layer = MultiMomdalTranfsformerDecoderlayer(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.decoder = MultiMomdalTranfsformerDecoder(decoder=decoder_layer,num_layers=num_layers)
        encoder_layer = DualChannelEncoderLayer(dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop,drop_path,dropout)
        self.encoder = DualChannelEncoder(dim,num_frames,encoder=encoder_layer,num_layers=num_layers)

    def forward(self,image,text,query):
        src = self.encoder(image,text)
        output = self.decoder(src,query)
        return output

# 多模态解码器
class MultiMomdalTranfsformerDecoder(nn.Module):
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


# 多模态解码器层
class MultiMomdalTranfsformerDecoderlayer(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.cross_attn = MultiHeadCrossAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.ffn = FFN(dim)
    def forward(self,src,tgt):
        tgt = tgt + self.self_attn(tgt)
        tgt = tgt + self.cross_attn(src,tgt)
        tgt = self.ffn(tgt)
        return tgt

# 多模态交互编码器
class DualChannelEncoder(nn.Module):
    def __init__(self,dim,num_frames,encoder=None,num_layers=6):
        super().__init__()
        assert encoder != None,"Encoder cannot be None"
        self.time_encod = TemporalEncoding(num_frames,d_model=dim)
        self.pos_encod = PositionalEncoding(d_model=dim)
        self.encoder_layers = self.get_clones(encoder,num_layers)
        self.linear1 = nn.Linear(dim,dim)
        self.linear2 = nn.Linear(dim,dim)

    def get_clones(self,module,num_layers):
        return ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

    def forward(self,image,text):
        image = self.time_encod(image)
        text = self.pos_encod(text)
        for model in self.encoder_layers:
            image,text = model(image,text)
        p = self.linear1(text)
        q = self.linear2(p)
        output = (1+q)*image+p
        return output

# 双通道编码器层
class DualChannelEncoderLayer(nn.Module):
    def __init__(self,dim,num_heads=8,qkv_bias=False,qk_scale=None,attn_drop=0.,proj_drop=0.,drop_path=0.1,dropout=0.1):
        super().__init__()
        self.text_attn = MultiHeadSelfAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.image_attn = MultiHeadTemporalSelfAttention(dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop,drop_path)
        self.T2I_attn = MultiHeadCrossAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.image_FFN = FFN(dim)
        self.text_FFN = FFN(dim)


    def forward(self,image,text):
        image = image + self.image_attn(image)
        text = text + self.text_attn(text)
        text = text + self.T2I_attn(image, text)
        image = self.image_FFN(image)
        text = self.text_FFN(text)
        return image,text




