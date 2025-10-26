import time
import torch
import datetime
from torch import nn
from transformers import CLIPVisionModel, CLIPTextModel, CLIPTokenizer
from models.module.Block import GroupWiseLinear
from utils.tools import AverageMeter
from models.module.Block import PositionalEncoding
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler
# from torch.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.amp import autocast

_MODELS = {
    'ViT-B/16': "openai/clip-vit-base-patch16",
    'ViT-B/32': "openai/clip-vit-base-patch32",
    'ViT-L/14': "openai/clip-vit-large-patch14"
}


class CLIP_Transformer(nn.Module):
    def __init__(
            self,
            class_embed,
            class_list,
            num_layers=6,
            image_mode="openai/clip-vit-base-patch16",
    ):
        super().__init__()
        self.num_classes, self.embed_dim = class_embed.shape
        self.query = class_embed
        self.class_list = class_list
        self.image_model = CLIPVisionModel.from_pretrained(image_mode)
        self.pos_encod = PositionalEncoding(d_model=self.embed_dim)
        self.linear1 = nn.Linear(in_features=self.image_model.config.hidden_size, out_features=self.embed_dim,
                                 bias=False)
        # pytorch comes with transformer
        self.transformer = nn.Transformer(num_encoder_layers=num_layers, num_decoder_layers=num_layers,
                                          d_model=self.embed_dim, batch_first=True)
        # Own transformer design
        # self.transformer = Transformer(num_layers=num_layers,dim=self.embed_dim)

    def forward(self, images):
        b, t, c, h, w = images.size()
        images = self.image_model(images.reshape(b * t, c, h, w))[1].reshape(b, t, -1)
        images = self.linear1(images)
        images = self.pos_encod(images)
        query_embed = self.query.unsqueeze(0).repeat(b, 1, 1)
        x = self.transformer(images, query_embed)
        x = x.sum(-1)
        return x


class CLIP_Transformer_Executor():
    def __init__(self, criterion, eval_metric, eval_metric_string, class_list, num_frames, working_dir, device,
                 args) -> None:
        self.start_epoch = 0
        self.best_prec = 0
        self.best_epoch = 0
        self.criterion = criterion.to(device)
        self.eval_metric = eval_metric.to(device)
        self.eval_metric_string = eval_metric_string
        self.class_list = class_list
        self.working_dir = working_dir
        self.device = device
        self.control = args.logging.control
        self.schedule_enable = args.optimizer.enable
        self.distributed = args.distributed
        self.zero = args.expt.zero
        self.few = args.expt.few
        self.epoch = args.iterate.epoch
        self.max_interval = args.iterate.max_interval
        self.print_freq = args.logging.print_freq
        self.eval_freq = args.logging.eval_freq
        self.image_mode = _MODELS[args.pretrain.image_mode]
        self.text_mode = _MODELS[args.pretrain.text_mode]
        self.tokenizer = CLIPTokenizer.from_pretrained(self.text_mode)
        self.mixup_enable = args.mixup.enable
        self.grad_clip_enable = args.grad_clip.enable
        self.grad_clip_mode = args.grad_clip.mode
        self.grad_clip_value = args.grad_clip.value
        self.scaler = GradScaler()
        if args.mixup.enable:
            from models.module.MixUp import MixUp
            self.mixup = MixUp(args.mixup.alpha)

        class_embed = self._get_text_features(class_list, self.text_mode)
        model = CLIP_Transformer(
            class_embed=class_embed.to(device),
            num_layers=args.transformer.num_layers,
            class_list=class_list,
            image_mode=self.image_mode
        ).to(device)
        for p in model.parameters():
            p.requires_grad = True
        for p in model.image_model.parameters():
            p.requires_grad = False
        if self.distributed:
            self.model = DDP(model, device_ids=[device], find_unused_parameters=True)
        else:
            self.model = model
        self.optimizer = Adam([{"params": self.model.parameters(), "lr": args.optimizer.lr}])
        if args.optimizer.enable:
            self.scheduler = CosineAnnealingWarmRestarts(self.optimizer, T_0=args.optimizer.T_0,
                                                         eta_min=args.optimizer.eta_min)

    @staticmethod
    def _get_prompt(cl_names):
        temp_prompt = []
        for c in cl_names:
            temp_prompt.append(c)
        return temp_prompt

    def _get_text_features(self, cl_names, text_mode="openai/clip-vit-base-patch32"):
        text_model = CLIPTextModel.from_pretrained(text_mode)
        tokenizer = CLIPTokenizer.from_pretrained(text_mode)
        act_prompt = self._get_prompt(cl_names)
        texts = tokenizer(act_prompt, padding=True, return_tensors="pt")
        text_class = text_model(**texts).pooler_output.detach()
        return text_class

    def run(self, train_loader, val_loader, logger):
        global best_prec
        global best_epoch
        best_epoch = self.best_epoch
        best_prec = self.best_prec
        for epoch in range(self.start_epoch, self.epoch):
            if self.distributed:
                train_loader.sampler.set_epoch(epoch)
            self.train(train_loader, epoch, logger)
            if self.schedule_enable:
                self.scheduler.step()  # 更新学习率
            if (epoch + 1) % self.eval_freq == 0:
                if self.control:
                    self.eval_freq = 1
                prec = self.valid(val_loader, logger)
                is_best = prec > best_prec
                if is_best:
                    best_prec = prec
                    best_epoch = epoch
                logger.info('Testing: {}/{}'.format(prec, best_prec))
                filename1 = "{}/model.pt".format(self.working_dir, epoch)
                self.save(epoch, best_prec, best_epoch, filename1)
                logger.info("Saving:{}".format(filename1))
                if is_best:
                    filename2 = "{}/best_model.pt".format(self.working_dir)
                    self.save(epoch, best_prec, best_epoch, filename2)
                    logger.info('Saving:{}'.format(filename2))
                if epoch - best_epoch > self.max_interval:
                    logger.info("The maximum epoch interval is exceeded")
                    break

    def train(self, train_loader, epoch, logger):

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        self.model.train()
        end = time.time()
        for i, (data, label, _) in enumerate(train_loader):
            data_time.update(time.time() - end)
            dimen = data.shape[0]
            data, label = data.to(self.device), label.to(self.device)
            if self.mixup_enable:
                # 应用MixUp
                data, label_a, label_b, lam = self.mixup(data, label)
            self.optimizer.zero_grad()
            with autocast(self.device):
                output = self.model(data)
                if self.mixup_enable:
                    loss = lam * self.criterion(output, label_a) + (1 - lam) * self.criterion(output, label_b)
                else:
                    loss = self.criterion(output, label)
            del output
            del data
            del label
            # 混合精度反向传播
            self.scaler.scale(loss).backward()
            if self.grad_clip_enable:
                if self.grad_clip_mode == "norm":
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_clip_value)
                elif self.grad_clip_mode == "clip":
                    torch.nn.utils.clip_grad_value_(self.model.parameters(), clip_value=self.grad_clip_value)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            losses.update(loss.item(), dimen)
            del loss

            batch_time.update(time.time() - end)
            cur_iter = epoch * len(train_loader) + i
            max_iter = self.epoch * len(train_loader)
            eta_sec = batch_time.avg * (max_iter - cur_iter + 1)
            eta_sec = str(datetime.timedelta(seconds=int(eta_sec)))

            if i % self.print_freq == 0:
                logger.info(('Epoch: [{0}][{1}/{2}], lr: {lr:.2e}, eta: {3}\t'
                             'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                             'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                             'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(
                    epoch + 1, i, len(train_loader), eta_sec, batch_time=batch_time, data_time=data_time, loss=losses,
                    lr=self.optimizer.param_groups[-1]['lr'])))

    def valid(self, val_loader, logger):
        eval_meter = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for i, (data, label, _) in enumerate(val_loader):
                dimen = data.shape[0]
                data, label = data.to(self.device), label.long().to(self.device)
                output = self.model(data)
                eval = self.eval_metric(output, label)
                del output
                del data
                del label
                eval_meter.update(eval.item(), dimen)
                del eval
                if i % self.print_freq == 0:
                    logger.info(
                        ('Test: [{0}/{1},{2}:{map:.3f}]\t'.format(i, len(val_loader), self.eval_metric_string,
                                                                  map=eval_meter.avg * 100)))
        logger.info(
            ('Testing Results {0} === {result:.3f}'.format(self.eval_metric_string, result=eval_meter.avg * 100)))
        return eval_meter.avg * 100

    def load(self, file_path):
        checkpoint = torch.load(file_path)
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_prec = checkpoint['best_prec']
        self.best_epoch = checkpoint['best_epoch']
        if isinstance(self.model, DDP):
            self.model.module.linear1.load_state_dict(checkpoint['linear1'])
            self.model.module.transformer.load_state_dict(checkpoint["transformer"])
        else:
            self.model.linear1.load_state_dict(checkpoint['linear1'])
            self.model.transformer.load_state_dict(checkpoint["transformer"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in self.optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(self.device)

    def save(self, epoch, best_prec, best_epoch, filename):
        if isinstance(self.model, DDP):
            torch.save({
                'epoch': epoch,
                'best_prec': best_prec,
                'best_epoch': best_epoch,
                "linear1": self.model.module.linear1.state_dict(),
                'transformer': self.model.module.transformer.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, filename)
        else:
            torch.save({
                'epoch': epoch,
                'best_prec': best_prec,
                'best_epoch': best_epoch,
                "linear1": self.model.linear1.state_dict(),
                'transformer': self.model.transformer.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, filename)
