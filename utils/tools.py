# @Time : 2024/1/17 20:13
# @Author : Zeyong Ji

import os
import socket
import random
import numpy as np
import torch
import configparser as cp
import datetime
import pytz

# Evaluation index calculation class
class AverageMeter(object):
    """Computes and stores the average and current value
       Imported from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L247-L262
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


# Get current time
def get_format_time(fmt=None):
    format = "%Y-%m-%d_%H-%M-%S"
    if fmt != None:
        format = fmt
    # Format the struct_time object to the specified time string using the strftime() method
    current_time = datetime.datetime.now(pytz.timezone('Asia/Shanghai')).strftime(format)
    # Output the current year, month, day, hour, second
    print('currnet time:',current_time)
    return current_time

# 固定随机种子
def fixed_random_seed(seed):
    if (seed >= 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        print("[INFO] Setting SEED: " + str(seed))
    else:
        print("[INFO] Setting SEED: None")

# Connection path, making it common to both win and linux
def connect_path(lists):
    path = ''
    for list in lists:
        path = os.path.join(path,list)
    return path.replace('\\','/')



# Obtain the environment parameters based on the device
def get_env_config():
    config = cp.ConfigParser()
    cur_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..')).replace('\\','/')
    print('current abs path:',cur_path)
    config.read(connect_path([cur_path,'location.cfg']))
    print('selectable sections:',config.sections())

    host = socket.gethostname() # Gets the name of the device (computer)
    print("currnet host name:",host)
    if host not in config.sections():
        host = 'kaggle'
    option = config.options(host)
    print('select device env:',host)
    print('attribute setting:',option)
    return config[host]


if __name__ == '__main__':
    get_env_config()
    get_format_time()


