import time
import torch
import datetime
import os.path
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from models.module.Executor import Executor
from models.module.MultiAddTransformer import MultiAddTransformer
from utils.tools import AverageMeter, connect_path
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.nn.functional as F

class CSGMARM(nn.Module):
    def __init__(self,
                 class_embed,
                 class_list,
                 num_frames=16,
                 num_layers=6,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 drop_path=0.1,
                 dropout=0.1,
                 image_mode="openai/clip-vit-base-patch16",
                 text_mode="openai/clip-vit-base-patch16",
                 ):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.query = class_embed
        self.class_list = class_list
        self.image_model = CLIPVisionModel.from_pretrained(image_mode)
        self.text_model = CLIPTextModel.from_pretrained(text_mode)
        self.linear1 = nn.Linear(in_features=self.image_model.config.hidden_size, out_features=self.embed_dim,bias=False)
        self.linear2 = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim, bias=False)
        self.transformer = MultiAddTransformer(self.embed_dim, num_frames, num_layers, num_heads, qkv_bias, qk_scale,
                                      attn_drop, proj_drop, drop_path, dropout)


    def forward(self, images, texts):
        b, t, c, h, w = images.size()
        images = self.linear1(self.image_model(images.reshape(b * t, c, h, w))[1].reshape(b, t, -1))
        texts = self.text_model(**texts).pooler_output.detach()
        texts = self.linear2(texts.reshape(b, t, -1))
        query_embed = self.query.unsqueeze(0).repeat(b, 1, 1)
        x = self.transformer(images, texts, query_embed)
        x = x.sum(-1)
        return x


class CSGMARM_Executor(Executor):
    def __init__(self, criterion, eval_metric, eval_metric_string, class_list, num_frames, working_dir,project_path, device,
                 args) -> None:
        super(CSGMARM_Executor, self).__init__(criterion, eval_metric, eval_metric_string, class_list, working_dir,project_path, device,args)
        class_embed = self._get_text_features(class_list, self.text_mode)
        model = CSGMARM(
            class_embed=class_embed.to(device),
            class_list=class_list,
            num_frames=num_frames,
            num_layers=args.transformer.num_layers,
            num_heads=args.transformer.num_heads,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=args.transformer.attn_drop,
            proj_drop=args.transformer.proj_drop,
            drop_path=args.transformer.drop_path,
            dropout=args.transformer.dropout,
            image_mode=self.image_mode,
            text_mode=self.text_mode,
        ).to(device)
        for p in model.parameters():
            p.requires_grad = True
        for p in model.image_model.parameters():
            p.requires_grad = False
        for p in model.text_model.parameters():
            p.requires_grad = False
        if self.distributed:
            self.model = DDP(model, device_ids=[device], find_unused_parameters=True)
        else:
            self.model = model
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": args.optimizer.lr}])
        if args.optimizer.enable:
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=args.optimizer.T_0,eta_min=args.optimizer.eta_min)
