import os
import numpy as np

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pprint
import sys
import argparse
import torch
import yaml
import shutil
import torchvision
from dotmap import DotMap
import torch.backends.cudnn as cudnn
from utils.tools import fixed_random_seed, get_env_config, connect_path, get_format_time
from utils.logger import setup_logger


def main(args):
    global best_prec
    global fixed_time
    fixed_time = get_format_time()

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config["resume"] == None:
        raise RuntimeError("No weight file found")

    # Obtain device environment parameters
    env = get_env_config()

    # Result output path:env path\results\[dataset name]\[current time]
    if config['expt']['zero'] == 'seen':
        working_dir = connect_path([
            env['output_path'], 'zero', config['data']['dataset'],
            config['expt']['zero'] + "_" + str(config['expt']['ratio']),
            config['iterate']['model'], config['pretrain']['image_mode'],
            fixed_time
        ])
    else:
        working_dir = connect_path([
            env['output_path'], 'results', 'zero', config['data']['dataset'],
            config['expt']['zero'], config['iterate']['model'],
            config['pretrain']['image_mode'], fixed_time
        ])

    # Read data path:
    data_dir = connect_path([env['input_path'], config['data']['dataset']])

    if not os.path.exists(data_dir):
        raise IOError("The dataset is not exist")

    if not os.path.exists(working_dir):
        os.makedirs(working_dir)
    shutil.copy(args.config, working_dir)

    # 构建logger，打印环境和配置文件信息
    logger = setup_logger(output=working_dir, distributed_rank=0, name=config['iterate']['model'])
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

    # Fixed random seed
    fixed_random_seed(config.seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:" + str(args.gpu)
        cudnn.deterministic = True
        cudnn.benchmark = True

    from dataload.datamanager import DataManager
    manager = DataManager(config, data_dir)
    class_list = list(manager.get_act_dict().keys())
    num_classes = len(class_list)

    # criterion or loss
    import torch.nn as nn
    if config.data.dataset in ['hmdb51', 'ucf101']:
        criterion = nn.CrossEntropyLoss()
    elif config.data.dataset in ['charades']:
        criterion = nn.BCEWithLogitsLoss()
    else:
        raise RuntimeError('Not dataset was selected')

    # evaluation metric
    if config.data.dataset in ['hmdb51', 'ucf101']:
        from torchmetrics.classification import MulticlassAccuracy
        eval_metric = MulticlassAccuracy(num_classes=num_classes, average='micro')
        eval_metric_string = 'Multiclass Accuracy'
    elif config.data.dataset in ['charades']:
        from torchmetrics.classification import MultilabelAveragePrecision
        eval_metric = MultilabelAveragePrecision(num_labels=num_classes, average='micro')
        eval_metric_string = 'Multilabel Average Precision'
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
            executor.weight_load(config.resume)
        else:
            logger.info("=> no checkpoint found at '{}'".format(config.resume))

    if config.expt.zero == "EP1":
        acc_list = []
        for i in range(config.expt.number):
            val_class, val_label = manager.get_data_class(num_classes // 2)
            val_transform = manager.get_transforms(mode='test')
            # Load data set
            val_loader = manager.get_zero_data_loader(val_transform, mode='test', label=val_label)
            logger.info("Test size:{}".format(len(val_loader.dataset)))
            logger.info("Select class total:{}".format(str(len(val_class))))
            logger.info("Thera are: {}".format(val_class))
            acc = executor.zero_valid(None, val_loader, logger)
            acc_list.append(acc)
            mean = np.mean(acc_list)
            std = np.std(acc_list)
            logger.info("The accuracy of the {} time is {:.3f}, the mean is {:.3f}, and the std is {:.3f}".format(i + 1, acc,mean, std))
    elif config.expt.zero == "EP2":
        # all data
        all_transforms = manager.get_transforms(mode='test')
        all_loader = manager.get_zero_data_loader(all_transforms)
        logger.info("All dataset size:{}".format(len(all_loader.dataset)))
        acc = executor.zero_valid(None, all_loader, logger)
        logger.info("The all dataset accuracy is {result:.3f}".format(result=acc))
    elif config.expt.zero == "EP3":
        acc_list = []
        for i in range(1,4):
            config.data.file_number = i
            manager = DataManager(config, data_dir)
            val_transform = manager.get_transforms(mode='test')
            # Load data set
            val_loader = manager.get_data_loader(val_transform, mode='test')
            logger.info("Test size:"+str(len(val_loader.dataset)))
            acc = executor.zero_valid(None,val_loader,logger)
            acc_list.append(acc)
            mean = np.mean(acc_list)
            std = np.std(acc_list)
            logger.info("The accuracy of the {} file is {:.3f}, the mean is {:.3f}, and the std is {:.3f}".format(i, acc,mean, std))
    elif config.expt.zero == "seen":
        # The seen and unseen classes are divided randomly and proportionally
        train_dict, test_dict = manager.get_split_categories(config.expt.ratio)
        logger.info("Seen dataset size:{}".format(len(train_dict)))
        logger.info("Seen class are:{}".format(list(train_dict.keys())))
        logger.info("Unseen dataset size:{}".format(str(len(test_dict))))
        logger.info("Unseen class are:{}".format(list(test_dict.keys())))
        # Load train
        train_transform = manager.get_transforms(mode='train')
        train_loader = manager.get_zero_data_loader(train_transform, mode='train', label=list(train_dict.values()))
        logger.info("Train size:{}".format(len(train_loader.dataset)))
        # Load test
        test_transform = manager.get_transforms(mode='test')
        test_loader = manager.get_zero_data_loader(test_transform, mode='test', label=list(test_dict.values()))
        logger.info("Test size:{}".format(len(test_loader.dataset)))
        executor.zero_valid(train_loader, test_loader, logger)

    logger.warning('Close log file: %s ' % (logger.name))
    logger.handlers.clear()


def get_parser():
    parser = argparse.ArgumentParser(description="Training script for action recognition")
    parser.add_argument('--config', '-cfg', type=str, default='./configs/hmdb51/hmdb51_vitb16_zero.yaml',help='global config file')
    # parser.add_argument('--config', '-cfg', type=str, default='./configs/ucf101/ucf101_vitb16_zero.yaml',help='global config file')
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_parser()
    main(args)
