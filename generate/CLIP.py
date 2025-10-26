# @Time : 2024/4/4 13:49
# @Author : Zeyong Ji
import argparse
import os
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pprint
import sys
from tqdm import tqdm
import pandas as pd
import torch
import torchvision
from utils.tools import get_env_config,connect_path,fixed_random_seed
from utils.logger import setup_logger
from transformers import CLIPProcessor, CLIPModel

_MODEL={
    'ViT-B/16':'openai/clip-vit-base-patch16',
    'ViT-B/32':'openai/clip-vit-base-patch32',
    'ViT-L/14':'openai/clip-vit-large-patch14'
}

def main(args):
    env = get_env_config()

    if args.dataset in ['ucf101','hmdb51']:
        doc_dir = connect_path([env['project_path'],'doc', 'CLIP', args.dataset, args.clip_mode,'top-{}'.format(args.top_k),str(args.file_number)])
    elif args.dataset in ['charades','animalkingdom','kinetics400']:
        doc_dir = connect_path([env['project_path'],'doc', 'CLIP', args.dataset, args.clip_mode,'top-{}'.format(args.top_k)])
    else:
        raise NotImplementedError
    data_dir = connect_path([env['input_path'], args.dataset])
    if not os.path.exists(doc_dir):
        os.makedirs(doc_dir)
    if not os.path.exists(data_dir):
        raise IOError("The dataset is not exist")
    logger = setup_logger(output=doc_dir, distributed_rank=0, name='CLIP Generator')
    logger.info("------------------------------------")
    logger.info("Environment Versions:")
    logger.info("- Python: {}".format(sys.version))
    logger.info("- PyTorch: {}".format(torch.__version__))
    logger.info("- TorchVison: {}".format(torchvision.__version__))
    logger.info("------------------------------------")
    pp = pprint.PrettyPrinter(indent=4)
    parser_dict = args.__dict__
    logger.info(pp.pformat(parser_dict))
    logger.info("------------------------------------")
    logger.info("doc path: {}".format(doc_dir))
    logger.info("data path: {}".format(data_dir))

    device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda"
    if torch.cuda.is_available():
        device = "cuda:"+str(args.gpu)

    fixed_random_seed(args.seed)

    template = "{}_frame_{}_{}_statement.txt"

    # CLIP
    model = CLIPModel.from_pretrained(_MODEL[args.clip_mode]).to(device)
    processor = CLIPProcessor.from_pretrained(_MODEL[args.clip_mode])

    # data = pd.read_csv(connect_path([env['project_path'],'doc','K400_labels.csv']))
    # text = data['label'].tolist()
    text = []
    path = connect_path([env['project_path'], 'doc', 'action_prompts.txt'])
    with open(path, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            text.append(line)
    logger.info("Text length:{}".format(len(text)))
    top = args.top_k
    # time.sleep(400)

    from generate_dataload.datamanager import DataManager
    manager = DataManager(args, data_dir)

    train_data = manager.get_dataset(mode=args.train_mode)
    train_size = train_data.get_len()
    logger.info("Train size:" + str(train_size))

    train_list = []
    for i in tqdm(range(train_size),desc="Generating train statement by CLIP"):
        img_list, _, _, index = train_data.get_item(i)
        inputs = processor(text=text,images=img_list,return_tensors='pt',padding=True).to(device)
        del img_list
        outputs = model(**inputs)
        del inputs
        logits_per_image = outputs.logits_per_image
        del outputs
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
        del logits_per_image
        for j in range(len(index)):
            a = probs[j].argsort()[-top:][::-1]
            abstract = ",".join([text[idx] for idx in a])
            abstract = args.prompt.format(abstract)
            abstract = str(index[j])+" "+abstract
            train_list.append(abstract)
        del probs

    train_list = '\n'.join(train_list)
    train_txt_path = connect_path([doc_dir, template.format(args.dataset, args.total_length, args.train_mode)])
    with open(train_txt_path, 'w') as f:
        f.writelines(train_list)
    logger.info("{} is save".format(train_txt_path))
    del train_list
    #
    test_data = manager.get_dataset(mode=args.val_mode)
    test_size = test_data.get_len()
    logger.info("Test size:" + str(test_size))

    test_list = []
    for i in tqdm(range(test_size), desc="Generating test statement by CLIP"):
        img_list, _, _, index = test_data.get_item(i)
        inputs = processor(text=text, images=img_list, return_tensors='pt', padding=True).to(device)
        del img_list
        outputs = model(**inputs)
        del inputs
        logits_per_image = outputs.logits_per_image
        del outputs
        probs = logits_per_image.softmax(dim=1).cpu().detach().numpy()
        del logits_per_image
        for j in range(len(index)):
            a = probs[j].argsort()[-top:][::-1]
            abstract = ",".join([text[idx] for idx in a])
            abstract = args.prompt.format(abstract)
            abstract = str(index[j]) + " " + abstract
            test_list.append(abstract)
        del probs

    test_list = '\n'.join(test_list)
    test_txt_path = connect_path([doc_dir, template.format(args.dataset, args.total_length, args.val_mode)])
    with open(test_txt_path, 'w') as f:
        f.writelines(test_list)
    logger.info("{} is save".format(test_txt_path))
    del test_list
    logger.warning('Close log file: %s ' % (logger.name))
    logger.handlers.clear()

def get_parser():
    parser = argparse.ArgumentParser(description='CLIP Generater Text Prompt')
    parser.add_argument("--desc",default="Using CLIP Text Prompt")
    parser.add_argument('--seed',default=2024,type=int,help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--dataset", default="ucf101", help="Dataset:ucf101, hmdb51,charades,animalkingdom,kinetics400")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument('--clip_mode',default="ViT-B/16",type=str,help="The selected image mode")
    parser.add_argument('--train_mode', default="train", type=str,help="The selected dataset mode")
    parser.add_argument('--val_mode', default="test", type=str,help="The selected dataset mode")
    parser.add_argument('--prompt', default="A video of {}", type=str,help="The selected image mode")
    parser.add_argument('--top_k',default=1,type=int,help="The top-k of action words")
    parser.add_argument('--file_number',default=1,type=int,help="select ucf101/hdmb51 split file")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--generate', type=str, default='CLIP', help="generate mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    # 16 frames
    # args.total_length = 16
    #
    # args.top_k = 9
    # args.dataset = 'charades'
    # main(args)
    # args.dataset = 'hmdb51'
    # main(args)
    #
    # args.top_k = 11
    # args.dataset = 'charades'
    # main(args)
    # args.dataset = 'hmdb51'
    # main(args)


    # 32 frames
    # args.total_length = 32
    # args.dataset = 'kinetics400'
    # args.top_k = 5
    # main(args)
    #
    #
    # args.dataset = 'hmdb51'
    # args.top_k = 5
    # args.file_number = 1
    # main(args)
    #
    # args.dataset = 'ucf101'
    # args.top_k = 5
    # args.file_number = 1
    # main(args)




