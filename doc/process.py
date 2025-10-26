import pandas as pd
import argparse
import os.path
from utils.tools import get_env_config, connect_path

def main(args):
    env = get_env_config()
    if args.dataset in ['ucf101', 'hmdb51']:
        test_txt = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
             str(args.file_number), "{}_frame_{}_test_statement.txt".format(args.dataset, args.total_length)])
        train_txt = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
             str(args.file_number), "{}_frame_{}_train_statement.txt".format(args.dataset, args.total_length)])
        full_text = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
             str(args.file_number), '{}_frame_{}_full_statement.txt'.format(args.dataset, args.total_length)])
        test_anno_file = connect_path([env['input_path'], args.dataset, 'annotations',
                                       '{}_test_split_{}_frames.txt'.format(args.dataset, args.file_number)])
        train_anno_file = connect_path([env['input_path'], args.dataset, 'annotations',
                                        '{}_train_split_{}_frames.txt'.format(args.dataset, args.file_number)])
        full_anno_file = connect_path(
            [env['input_path'], args.dataset, 'annotations', '{}_full_frames.txt'.format(args.dataset)])
    elif args.dataset in ['charades']:
        test_txt = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
            "{}_frame_{}_test_statement.txt".format(args.dataset, args.total_length)])
        train_txt = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
             "{}_frame_{}_train_statement.txt".format(args.dataset, args.total_length)])
        full_text = connect_path(
            [env['project_path'], 'doc', 'CLIP', args.dataset, args.clip_mode, 'top-{}'.format(args.top_k),
             '{}_frame_{}_full_statement.txt'.format(args.dataset, args.total_length)])
        test_anno_file = connect_path([env['input_path'], args.dataset, 'annotations',
                                       'Charades_v1_test.csv'])
        train_anno_file = connect_path([env['input_path'], args.dataset, 'annotations',
                                        'Charades_v1_train.csv'])
        full_anno_file = connect_path(
            [env['input_path'], args.dataset, 'annotations', 'Charades_v1_full.csv'])

    else:
        raise NotImplementedError

    print(test_txt)
    print(train_txt)
    print(full_text)
    print(test_anno_file)
    print(train_anno_file)
    print(full_anno_file)

    if not (os.path.exists(test_txt) and os.path.exists(train_txt) and os.path.exists(
            test_anno_file) and os.path.exists(train_anno_file)):
        raise IOError("The txt is not exist")


    if args.dataset in ['ucf101', 'hmdb51']:
        all_list = []
        all_statement_list = []
        num = 0
        with open(train_anno_file, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                all_list.append(line)

        with open(train_txt, 'r') as f:
            lines = f.read().splitlines()
            statement_list = []
            for line in lines:
                _, statement = line.split(" ", 1)
                statement_list.append(statement)
                num = num + 1
                if num % args.total_length == 0:
                    all_statement_list.append(statement_list)
                    statement_list = []

        with open(test_anno_file, 'r') as f:
            lines = f.read().splitlines()
            for line in lines:
                all_list.append(line)

        with open(test_txt, 'r') as f:
            lines = f.read().splitlines()
            statement_list = []
            for line in lines:
                _, statement = line.split(" ", 1)
                statement_list.append(statement)
                num = num + 1
                if num % args.total_length == 0:
                    all_statement_list.append(statement_list)
                    statement_list = []

        combine = zip(all_list, all_statement_list)
        sorted_combined = sorted(combine)

        # 分离排序后的列表
        sorted_list1 = [x[0] for x in sorted_combined]
        sorted_list2 = [x[1] for x in sorted_combined]
        sorted_list2 = [x1 for x in sorted_list2 for x1 in x]
        for i, j in enumerate(sorted_list2):
            sorted_list2[i] = str(i) + " " + j

        sorted_list1 = "\n".join(sorted_list1)
        sorted_list2 = "\n".join(sorted_list2)
        with open(full_anno_file, 'w') as f:
            f.writelines(sorted_list1)
        print("All dataset sample is save")
        with open(full_text, 'w') as f:
            f.writelines(sorted_list2)
        print("All text promtp is save")
    elif args.dataset in ['charades']:
        try:
            # 读取第一个 CSV 文件，保留表头
            df1 = pd.read_csv(train_anno_file)
            # 读取第二个 CSV 文件，跳过表头
            df2 = pd.read_csv(test_anno_file, header=None, skiprows=1)
            # 确保第二个 DataFrame 的列名与第一个相同
            df2.columns = df1.columns
            # 拼接两个 DataFrame
            combined_df = pd.concat([df1, df2], ignore_index=True)
            # 将拼接后的 DataFrame 保存为新的 CSV 文件
            combined_df.to_csv(full_anno_file, index=False)
            print("All dataset sample is save")
        except FileNotFoundError:
            print("Error: The specified CSV file was not found.")
        except Exception as e:
            print(f"Error: An unknown error occurred: {e}")

        # text prompt
        index_list = []
        statement_list = []
        num = 0
        with open(train_txt, "r") as f:
            for line in f.readlines():
                _, statement = line.strip("\n").split(" ", 1)
                index_list.append(num)
                statement_list.append(statement)
                num = num + 1
        with open(test_txt, "r") as f:
            for line in f.readlines():
                _, statement = line.strip("\n").split(" ", 1)
                index_list.append(num)
                statement_list.append(statement)
                num = num + 1

        combine = zip(index_list, statement_list)
        combine = sorted(combine)

        sort_list = [str(x[0]) + " " + x[1] for x in combine]
        sort_list = "\n".join(sort_list)
        with open(full_text, "w") as f:
            f.writelines(sort_list)
        print("All text promtp is save")


def get_parser():
    parser = argparse.ArgumentParser(description='combine text prompt ')
    parser.add_argument("--desc", default="Using CLIP generate abstract")
    parser.add_argument('--seed', default=2024, type=int, help="Seed for Numpy and PyTorch. Default: -1 (None)")
    parser.add_argument("--dataset", default="charades", help="Dataset:ucf101, hmdb51,charades")
    parser.add_argument("--total_length", default=16, type=int, help="Number of frames in a video")
    parser.add_argument('--clip_mode', default="ViT-B/16", type=str, help="The selected image mode")
    parser.add_argument('--train_mode', default="train", type=str, help="The selected dataset mode")
    parser.add_argument('--val_mode', default="test", type=str, help="The selected dataset mode")
    parser.add_argument('--prompt', default="A video of {}", type=str, help="The selected image mode")
    parser.add_argument('--top_k', default=5, type=int, help="The top-k of text")
    parser.add_argument('--file_number', default=1, type=int, help="select ucf101/hdmb51 split file")
    parser.add_argument('--gpu', type=int, default=0, help='gpu id')
    parser.add_argument('--generate', type=str, default='CLIP', help="generate mode")
    return parser.parse_args()


if __name__ == '__main__':
    args = get_parser()
    main(args)
