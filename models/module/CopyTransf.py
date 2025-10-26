import copy
import torch
from torch import nn
from torch.nn import  ModuleList
from models.module.Block import MultiHeadSelfAttention,MultiHeadCrossAttention,MultiHeadTemporalSelfAttention,FFN,TemporalEncoding,PositionalEncoding



class CopyTransf(nn.Module):
    def __init__(self,
                 dim,num_frames=16,
                 num_layers=6,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 dropout=0.1,
                 fusion="MultiAdd",
                 strategy="singleText",
                 ):
        super().__init__()
        encoder_layer = DualChannelEncoderLayer(dim, num_heads, qkv_bias, qk_scale, attn_drop, proj_drop, drop_path,dropout,strategy=strategy)
        self.encoder = DualChannelEncoder(dim, num_frames, encoder=encoder_layer, num_layers=num_layers,fusion=fusion)
        decoder_layer = MultiMomdalTranfsformerDecoderlayer(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.decoder = MultiMomdalTranfsformerDecoder(decoder=decoder_layer,num_layers=num_layers)


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

# 双通道编码器
class DualChannelEncoder(nn.Module):
    def __init__(self,dim,num_frames,encoder=None,num_layers=6,fusion="MultiAdd"):
        super().__init__()
        assert encoder != None,"Encoder cannot be None"
        self.time_encod = TemporalEncoding(num_frames,d_model=dim)
        self.pos_encod = PositionalEncoding(d_model=dim)
        self.encoder_layers = self.get_clones(encoder,num_layers)
        self.linear1 = nn.Linear(dim,dim)
        self.linear2 = nn.Linear(dim,dim)
        self.fusion_module = Fusion(dim,fusion)

    def get_clones(self,module,num_layers):
        return ModuleList([copy.deepcopy(module) for _ in range(num_layers)])

    def forward(self,image,text):
        image = self.time_encod(image)
        text = self.pos_encod(text)
        for model in self.encoder_layers:
            image,text = model(image,text)
        output = self.fusion_module(image,text)
        return output

# 融合模块
class Fusion(nn.Module):
    def __init__(self,dim,fusion="MultiAdd"):
        # fusion = ["MultiAdd","AddMulti","CrossModelGRU","SpatioSemanticAttn"]
        super().__init__()
        self.fusion = fusion
        if self.fusion == "MultiAdd" or self.fusion == "AddMulti":
            self.linear1 = nn.Linear(dim, dim)
            self.linear2 = nn.Linear(dim, dim)
        elif self.fusion == "CrossModelGRU":
            self.linear = nn.Linear(2*dim,dim)
            self.sigmoid = nn.Sigmoid()
        elif self.fusion == "SpatioSemanticAttn":
            self.spatial = nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0)
            self.softmax = nn.Softmax(dim=-1)

    def forward(self,image,text):
        output = None
        if self.fusion == "MultiAdd":
            p = self.linear1(text)
            q = self.linear2(p)
            output = (1 + p) * image + q
        elif self.fusion == "AddMulti":
            p = self.linear1(text)
            q = self.linear2(p)
            output = (1 + q) * image + p
        elif self.fusion == "CrossModelGRU":
            gru = self.sigmoid(self.linear(torch.concat((image,text),dim=-1)))
            output = gru * image + (1 - gru) * text
        elif self.fusion == "SpatioSemanticAttn":
            b,t,d = image.shape
            h= w = d//2
            spatial = image.reshape(b, t, w, h )
            spatial_attn = self.spatial(spatial)
            spatial_attn = spatial_attn.reshape(b, t, d)
            # 处理语义注意力部分
            # 计算图像特征和文本特征的相似度
            similarity = torch.bmm(image, text.permute(0, 2, 1))  # (32, 16, 57)
            # 实例化 Softmax 并在最后一个维度上进行操作
            semantic_attn = self.softmax(similarity)
            # 对 semantic_attn 进行调整以匹配 spatial_attn 和 image 的形状
            # 这里假设我们取语义注意力的加权和
            semantic_attn = torch.bmm(semantic_attn, text)  # (32, 16, 256)
            # 融合
            output = spatial_attn * semantic_attn * image
        return output


# 双通道编码器层
class DualChannelEncoderLayer(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 dropout=0.1,
                 strategy="singleText"):
        # strategy:[parallel,serialImageText,serialTextImage,singleText,singleImage]
        super().__init__()
        self.parallel = strategy
        self.text_attn = MultiHeadSelfAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        self.image_attn = MultiHeadTemporalSelfAttention(dim,num_heads,qkv_bias,qk_scale,attn_drop,proj_drop,drop_path)
        if self.parallel == "singleText":
            self.T2I_attn = MultiHeadCrossAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        elif self.parallel == "singleImage":
            self.I2T_attn = MultiHeadCrossAttention(dim,num_heads,qkv_bias,qk_scale,dropout)
        else:
            self.T2I_attn = MultiHeadCrossAttention(dim, num_heads, qkv_bias, qk_scale, dropout)
            self.I2T_attn = MultiHeadCrossAttention(dim, num_heads, qkv_bias, qk_scale, dropout)
        self.image_FFN = FFN(dim)
        self.text_FFN = FFN(dim)


    def forward(self,image,text):
        image = image + self.image_attn(image)
        text = text + self.text_attn(text)
        # 单文本
        if self.parallel == "singleText":
            text = text + self.T2I_attn(image, text)
        # 单图像
        elif self.parallel == "singleImage":
            image = image + self.I2T_attn(text,image)
        # 并行策略
        elif self.parallel == "parallel":
            image_new = image + self.I2T_attn(text,image)
            text_new = text + self.T2I_attn(image, text)
            image = image_new
            text = text_new
        # 先图像后文本
        elif self.parallel == "serialImageText":
            image = image + self.I2T_attn(text,image)
            text = text + self.T2I_attn(image, text)
        # 先文本后图像
        elif self.parallel == "serialTextImage":
            text = text + self.T2I_attn(image, text)
            image = image + self.I2T_attn(text, image)
        image = self.image_FFN(image)
        text = self.text_FFN(text)
        return image,text




