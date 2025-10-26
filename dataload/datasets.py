import itertools
import os
import csv
import glob
from torch.utils import data
from itertools import compress
from utils.tools import connect_path
from .transforms_ss import *


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return self._data[2]


# 行为数据类（父类）
class ActionDataset(data.Dataset):
    def __init__(self, total_length):
        self.total_length = total_length
        self.video_list = []
        self.file_list = []
        self.index_list = []
        self.random_shift = False

    def _sample_indices(self, num_frames):
        if num_frames <= self.total_length:
            indices = np.linspace(0, num_frames - 1, self.total_length, dtype=int)
        else:
            ticks = np.linspace(0, num_frames, self.total_length + 1, dtype=int)
            if self.random_shift:
                indices = ticks[:-1] + np.random.randint(ticks[1:] - ticks[:-1])
            else:
                indices = ticks[:-1] + (ticks[1:] - ticks[:-1]) // 2
        return indices

    @staticmethod
    def _load_image(directory, image_name):
        return [Image.open(connect_path([directory, image_name])).convert('RGB')]

    def __getitem__(self, index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        indexs = self.index_list[index]
        indices = self._sample_indices(record.num_frames)
        return self._get(record, image_names, indexs, indices)

    def __len__(self):
        return len(self.video_list)


# AnimalKingdom data loading class
class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path, 'annotations', mode + '.csv'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list, self.index_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        index_list = []
        index = 0
        with open(self.anno_path,encoding="ISO-8859-1") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                # local
                path = connect_path([self.path, 'frames', ovid])
                # kaggle
                # path = connect_path([self.path, 'frames','frames', ovid])
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
                index_list += [list(range(index, index + self.total_length))]
                index += self.total_length
        return video_list, file_list, index_list

    def _get(self, record, image_names, indexs, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except:
                print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        indexs = np.asarray(indexs, dtype=np.integer)
        return process_data, label, indexs


# Charades data loading class
class Charades(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train', zero=None,
                 label=None):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.zero = zero
        self.label = label
        if zero == 'seen':
            self.anno_path = connect_path([self.path, 'annotations', 'Charades_v1_full.csv'])
        else:
            self.anno_path = connect_path([self.path, 'annotations', 'Charades_v1_' + mode + '.csv'])

        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list, self.index_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        video_list = []
        file_list = []
        index_list = []
        index = 0
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row['actions']
                if actions == '': continue
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in
                           [a.split(' ') for a in actions.split(';')]]
                if self.zero == 'seen':
                    for a in actions:
                        if a[0] not in self.zero_label:
                            index += self.total_length
                            continue
                vid = row['id']
                # kaggle
                # path = connect_path([self.path, 'frames','frames', vid])  # 读取RBG
                # local
                path = connect_path([self.path, 'frames', vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.num_classes), dtype=bool)
                for frame in range(num_frames):
                    for ann in actions:
                        if frame / fps > ann[1] and frame / fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                video_list += [VideoRecord([path, num_frames, labels[idx]])]
                index_list += [list(range(index, index + self.total_length))]
                index += self.total_length
        return video_list, file_list, index_list

    def _get(self, record, image_names, indexs, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        label = record.label[indices].any(0).astype(np.float32)
        indexs = np.asarray(indexs, dtype=np.integer)
        return process_data, label, indexs


# kinetics400 data loading class
class kinetics400(ActionDataset):
    def __init__(self, path, act_dict, total_length=16, transform=None, random_shift=False, mode='train'):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path, 'annotations', 'kinetics400_rgb_' + mode + '.txt'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list, self.index_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        index_list = []
        index = 0
        with open(self.anno_path, 'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                path = connect_path([self.path, 'frames', self.mode, vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                if num_frames < self.total_length: continue
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                index_list += [list(range(index, index + self.total_length))]
                index += self.total_length
        return video_list, file_list, index_list

    def _get(self, record, image_names, indexs, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        indexs = np.asarray(indexs, dtype=np.integer)
        return process_data, record.label, indexs


# HMDB51 data loading class
class HMDB51(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, file_number=1, transform=None, random_shift=False, mode='train',zero=None, few=None, label=None):
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.zero = zero
        self.few = few
        self.label=label
        if zero == "EP2" or zero == "seen":
            self.anno_path = connect_path( [self.path, 'annotations', 'hmdb51_full_frames.txt'])
        else:
            self.anno_path = connect_path([self.path, 'annotations', 'hmdb51_' + mode + '_split_' + str(file_number) + '_frames.txt'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list, self.index_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        index_list = []
        index = 0
        with open(self.anno_path, 'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                if self.zero == "EP1" or self.zero == "seen":
                    if int(labels) not in self.label:
                        index += self.total_length
                        continue
                path = connect_path([self.path, 'frames', vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                index_list += [list(range(index, index + self.total_length))]
                index += self.total_length
        if self.few is not None and self.mode == "train":
            video_list, file_list, index_list = self._few_process(video_list, file_list, index_list)
        return video_list, file_list, index_list

    def _few_process(self,video_list, file_list, index_list):
        # 使用 zip 函数将三个列表组合成一个包含元组的列表
        combined_list = list(zip(video_list, file_list, index_list))
        # 定义一个键函数，用于从组合列表的每个元组中提取第一个列表元素的指定索引处的值作为分组依据
        key_func = lambda x: x[0].label
        # 对组合列表进行排序，因为 itertools.groupby 要求输入是已排序的
        sorted_list = sorted(combined_list, key=key_func)
        result = []
        # 使用 itertools.groupby 按照键函数进行分组
        for _, group in itertools.groupby(sorted_list, key=key_func):
            group = list(group)
            # 如果组内元素数量大于等于 k，则从组中随机抽取 k 个元素
            if len(group) >= self.few:
                sampled = random.sample(group, self.few)
            else:
                # 如果组内元素数量小于 k，则将整个组添加到结果中
                sampled = group
            result.extend(sampled)
        # 拆分结果列表为三个独立的列表
        result_list1, result_list2, result_list3 = zip(*result)
        return list(result_list1), list(result_list2), list(result_list3)



    def _get(self, record, image_names, indexs, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        indexs = np.asarray(indexs, dtype=np.integer)
        return process_data, record.label, indexs


# ucf101 data loading class
class UCF101(ActionDataset):
    def __init__(self, path, act_dict, total_length=12, file_number=1, transform=None, random_shift=False,
                 mode='train',zero=None, few=None, label=None):
        super().__init__(total_length)
        self.path = path
        self.total_length = total_length
        self.transform = transform
        self.random_shift = random_shift
        self.mode = mode
        self.zero = zero
        self.few = few
        self.label = label
        if zero == "EP2":
                self.anno_path = connect_path( [self.path, 'annotations', 'ucf101_full_frames.txt'])
        else:
            self.anno_path = connect_path([self.path, 'annotations', 'ucf101_' + mode + '_split_' + str(file_number) + '_frames.txt'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        try:
            self.video_list, self.file_list, self.index_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _few_process(self,video_list, file_list, index_list):
        # 使用 zip 函数将三个列表组合成一个包含元组的列表
        combined_list = list(zip(video_list, file_list, index_list))
        # 定义一个键函数，用于从组合列表的每个元组中提取第一个列表元素的指定索引处的值作为分组依据
        key_func = lambda x: x[0].label
        # 对组合列表进行排序，因为 itertools.groupby 要求输入是已排序的
        sorted_list = sorted(combined_list, key=key_func)
        result = []
        # 使用 itertools.groupby 按照键函数进行分组
        for _, group in itertools.groupby(sorted_list, key=key_func):
            group = list(group)
            # 如果组内元素数量大于等于 k，则从组中随机抽取 k 个元素
            if len(group) >= self.few:
                sampled = random.sample(group, self.few)
            else:
                # 如果组内元素数量小于 k，则将整个组添加到结果中
                sampled = group
            result.extend(sampled)
        # 拆分结果列表为三个独立的列表
        result_list1, result_list2, result_list3 = zip(*result)
        return list(result_list1), list(result_list2), list(result_list3)

    def _parse_annotations(self):
        video_list = []
        file_list = []
        index_list = []
        index = 0
        with open(self.anno_path, 'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                if self.zero == "EP1":
                    if int(labels) not in self.label:
                        index += self.total_length
                        continue
                path = connect_path([self.path, 'frames', vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                index_list += [list(range(index, index + self.total_length))]
                index += self.total_length
        if self.few is not None and self.mode == "train":
            video_list, file_list, index_list = self._few_process(video_list, file_list, index_list)
        return video_list, file_list, index_list

    def _get(self, record, image_names, indexs, indices):
        images = list()
        for idx in indices:
            try:
                img = self._load_image(record.path, image_names[idx])
            except OSError:
                print('ERROR: Could not read image "{}"'.format(record.path))
                print('invalid indices: {}'.format(indices))
                raise
            images.extend(img)
        process_data = self.transform(images)
        process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
        indexs = np.asarray(indexs, dtype=np.integer)
        return process_data, record.label, indexs

# Thumos14 data loading class
# class Thumos14(ActionDataset):
#     def __init__(self, path, act_dict, total_length=12, transform=None, random_shift=False, mode='train'):
#         self.path = path
#         self.total_length = total_length
#         self.transform = transform
#         self.random_shift = random_shift
#         self.mode = mode if mode == 'test' else 'val'
#         self.act_dict = act_dict
#         self.num_classes = len(act_dict)
#         self.anno_path = connect_path([self.path, 'annotations', mode])
#         try:
#             self.video_list, self.file_list, self.index_list = self._parse_annotations()
#         except OSError:
#             print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
#             raise
#
#
#     def _parse_annotations(self):
#         path_frames = connect_path([self.path, 'frames', self.mode])
#         paths_videos = sorted(glob.glob(connect_path([path_frames,'*'])))
#         # consider the fps from the meta data
#         from scipy.io import loadmat
#         if self.mode == 'val':
#             file_meta_data = connect_path([self.anno_path, 'validation_set.mat'])
#             meta_key = 'validation_videos'
#         elif self.mode == 'test':
#             file_meta_data = connect_path([self.anno_path, 'test_set_meta.mat'])
#             meta_key = 'test_videos'
#         fps = loadmat(file_meta_data)[meta_key][0]['frame_rate_FPS'].astype(int)
#
#         video_fps = {}
#         video_frames = {}
#         video_num_frames = {}
#         for i, path in enumerate(paths_videos):
#             vid = path.split('/')[-1]
#             files = sorted(os.listdir(path))
#             video_fps[vid] = fps[i]
#             video_frames[vid] = files
#             num_frames = len(files)
#             video_num_frames[vid] = num_frames
#         file_list = []
#         video_list = []
#         index_list = []
#         index = 0
#         for cls in self.act_dict.keys():
#             path_ants_cls = connect_path([self.anno_path, cls + '_' + self.mode + '.txt'])
#             with open(path_ants_cls, 'r') as f:
#                 lines = f.read().splitlines()
#                 for lin in lines:
#                     vid, _, strt_sec, end_sec = lin.split(' ')
#                     strt_frme = np.ceil(float(strt_sec) * video_fps[vid]).astype(int)
#                     end_frme = np.floor(float(end_sec) * video_fps[vid]).astype(int)
#                     frames_ = video_frames[vid][strt_frme:end_frme + 1]
#                     num_frames = end_frme - strt_frme + 1
#                     if len(frames_) != num_frames:
#                         continue
#                         # breakpoint()
#                     file_list += [frames_]
#                     path = connect_path([path_frames, vid])
#                     video_list += [VideoRecord([path, num_frames, self.act_dict.get(cls)])]
#                     index_list += [list(range(index, index + self.total_length))]
#                     index += self.total_length
#         return video_list, file_list,index_list
#
#
#
#     def _get(self, record, image_names, indexs,indices):
#         images = list()
#         for idx in indices:
#             try:
#                 img = self._load_image(record.path, image_names[idx])
#             except OSError:
#                 print('ERROR: Could not read image "{}"'.format(record.path))
#                 print('invalid indices: {}'.format(indices))
#                 raise
#             images.extend(img)
#         process_data = self.transform(images)
#         process_data = process_data.view((self.total_length, -1) + process_data.size()[-2:])
#         indexs = np.asarray(indexs, dtype=np.integer)
#         return process_data, record.label,indexs
