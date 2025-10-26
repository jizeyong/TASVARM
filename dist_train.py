# @Time : 2025/1/25 8:37
# @Author : Zeyong Ji
import os
import yaml
import pprint
import sys
import argparse
import shutil
from dotmap import DotMap
import torch
import torchvision
import torch.backends.cudnn as cudnn
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
from utils.tools import fixed_random_seed, get_env_config, connect_path, get_format_time
from utils.logger import setup_logger

# 设置镜像路径 China
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"


def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    torch.cuda.set_device(rank)
    init_process_group(backend='nccl', rank=rank, world_size=world_size)


def main(rank: int, world_size: int, args: object):
    ddp_setup(rank, world_size)

    # 设置全局变量
    global best_prec
    global fixed_time

    # 设置当前时间戳
    fixed_time = get_format_time()

    # 加载yaml文件
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # 根据设备获取相关路径信息
    # Obtain device environment parameters
    env = get_env_config()

    # 构造路径
    # Result output path:env path\results\[dataset name]\[current time]
    working_dir = connect_path([
        env['output_path'], config['data']['dataset'], config['iterate']['model'],
        config['pretrain']['image_mode'], fixed_time
    ])
    # Read data path:
    data_dir = connect_path([env['input_path'], config['data']['dataset']])


    if not os.path.exists(data_dir):
            raise IOError("The dataset is not exist")

    if rank == 0:
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)
        shutil.copy(args.config, working_dir)

    # 构建logger，打印环境和配置文件信息
    logger = setup_logger(output=working_dir, distributed_rank=rank, name=config['iterate']['model'])
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    logger.info(pp.pformat(config))
    logger.info("------------------------------------")
    logger.info("results path: {}".format(working_dir))
    logger.info("data path: {}".format(data_dir))

    # 将配置文件构造成映射操作类
    config = DotMap(config)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise RuntimeError("CUDA is not available.")

    # Fixed random seed
    fixed_random_seed(config.seed)

    # Load data set
    from dataload.datamanager import DataManager
    manager = DataManager(config, data_dir)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # training data
    train_transform = manager.get_transforms(mode='train')
    train_loader = manager.get_data_loader(train_transform, mode='train')
    logger.info("Train size:" + str(len(train_loader.dataset)))

    # val or test data
    val_transform = manager.get_transforms(mode='test')
    val_loader = manager.get_data_loader(val_transform, mode='test')
    logger.info("Test size:" + str(len(val_loader.dataset)))

    # criterion or loss
    import torch.nn as nn
    if config.data.dataset in ['charades', 'animalkingdom']:
        criterion = nn.BCEWithLogitsLoss()
    elif config.data.dataset in ['hmdb51', 'ucf101', 'kinetics400']:
        criterion = nn.CrossEntropyLoss()

    # evaluation metric
    if config.data.dataset in ['charades', 'animalkingdom']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'
    elif config.data.dataset in ['hmdb51', 'ucf101', 'kinetics400']:
        from torchmetrics.classification import MulticlassAccuracy
        eval_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
        eval_metric_string = 'Multiclass Accuracy'
    else:
        raise RuntimeError('Not metric was selected')
    num_frames = config.iterate.total_length
    model_args = (criterion, eval_metric, eval_metric_string, class_list, num_frames, working_dir,env['project_path'],device, config)

    if config.iterate.model == 'CSGMARM':
        from models.TASVARM import CSGMARM_Executor
        executor = CSGMARM_Executor(*model_args)

    if config.resume:
        if os.path.isfile(config.resume):
            logger.info("=> loading checkpoint '{}'".format(config.resume))
            executor.load(config.resume)
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))
    executor.run(train_loader, val_loader, logger)
    logger.warning('Close log file: %s ' % (logger.name))
    logger.handlers.clear()
    destroy_process_group()


def get_parser():
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument('--config', '-cfg', type=str, default='./configs/hmdb51/hmdb51.yaml', help='global config file')
    parser.add_argument("--cuda_visible_devices", '-cvd', type=str, default='0,1', help='Specifies the idle GPU')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda_visible_devices
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
