# @Time : 2024/4/9 8:12
# @Author : Zeyong Ji
import csv
import os
import numpy as np
from PIL import Image
from itertools import compress
from utils.tools import connect_path

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

class ActionDataset():
    def __init__(self,total_length):
        self.total_length = total_length
        self.video_list = []
        self.file_list = []
        self.path_list = []
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


    def get_item(self,index):
        record = self.video_list[index]
        image_names = self.file_list[index]
        path = self.path_list[index]
        indices = self._sample_indices(record.num_frames)
        idxs = index * self.total_length
        return self._get(record,image_names,indices,path,idxs)


    def get_len(self):
        return len(self.video_list)

# AnimalKingdom data loading class
class AnimalKingdom(ActionDataset):
    def __init__(self, path, act_dict, total_length=12,random_shift=False, mode='train',generate="CLIP"):
        super().__init__(total_length)
        self.path = path
        self.total_length = total_length
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path, 'annotations', mode + '.csv'])
        # self.video_path = connect_path([self.path,'action_recognition','dataset','video'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self.generate = generate
        try:
            self.video_list, self.file_list,self.path_list= self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        path_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                ovid = row['video_id']
                labels = row['labels']
                path = connect_path([self.path, 'frames', ovid])
                files = sorted(os.listdir(path))
                file_list += [files]
                count = len(files)
                labels = [int(l) for l in labels.split(',')]
                video_list += [VideoRecord([path, count, labels])]
                # path = connect_path([self.video_path,ovid]) + '.mp4'
                path_list.append(path)
        return video_list, file_list,path_list

    def _get(self, record,image_names,indices,path,idxs):
        images = list()
        for idx in indices:
            if self.generate == "CLIP":
                try:
                    img = self._load_image(record.path, image_names[idx])
                except:
                    print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                    print('invalid indices: {}'.format(indices))
                    raise
            elif self.generate == "Qwen2-VL":
                img = [connect_path([record.path, image_names[idx]])]
            images.extend(img)
        label = np.zeros(self.num_classes)  # need to fix this hard number
        label[record.label] = 1.0
        index = np.arange(idxs, idxs + self.total_length, 1, dtype=np.integer)
        return images, label,path,index


# Charades data loading class
class Charades(ActionDataset):
    def __init__(self, path, act_dict, total_length=12,random_shift=False,mode='train',generate="CLIP"):
        super().__init__(total_length)
        self.path = path
        self.total_length = total_length
        self.mode = mode
        self.random_shift = random_shift
        self.anno_path = connect_path([self.path, 'annotations','Charades_v1_' + mode + '.csv'])
        # self.video_path = connect_path([self.path,'Charades_v1_480'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self.generate = generate
        try:
            self.video_list, self.file_list,self.path_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    @staticmethod
    def _cls2int(x):
        return int(x[1:])

    def _parse_annotations(self):
        video_list = []
        file_list = []
        path_list = []
        with open(self.anno_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                actions = row['actions']
                if actions == '': continue
                vid = row['id']
                path = connect_path([self.path,'frames', vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                fps = num_frames / float(row['length'])
                labels = np.zeros((num_frames, self.num_classes), dtype=bool)
                actions = [[self._cls2int(c), float(s), float(e)] for c, s, e in [a.split(' ') for a in actions.split(';')]]
                for frame in range(num_frames):
                    for ann in actions:
                        if frame/fps > ann[1] and frame/fps < ann[2]: labels[frame, ann[0]] = 1
                idx = labels.any(1)
                num_frames = idx.sum()
                file_list += [list(compress(files, idx.tolist()))]
                video_list += [VideoRecord([path, num_frames, labels[idx]])]
                # path = connect_path([self.video_path,vid])+'.mp4'
                path_list.append(path)
        return video_list, file_list,path_list

    def _get(self, record,image_names,indices,path,idxs):
        images = list()
        for idx in indices:
            if self.generate == "CLIP":
                try:
                    img = self._load_image(record.path, image_names[idx])
                except:
                    print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                    print('invalid indices: {}'.format(indices))
                    raise
            elif self.generate == "Qwen2-VL":
                img = [connect_path([record.path, image_names[idx]])]
            images.extend(img)
        label = record.label[indices].any(0).astype(np.float32)
        index = np.arange(idxs, idxs + self.total_length, 1, dtype=np.integer)
        return images, label, path, index

# kinetics400 data loading class
class kinetics400(ActionDataset):
    def __init__(self, path, act_dict, total_length=16, random_shift=False, mode='train',generate="CLIP"):
        self.path = path
        self.total_length = total_length
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path, 'annotations','kinetics400_rgb_' + mode + '.txt'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self.generate = generate
        try:
            self.video_list, self.file_list,self.path_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise
    def _parse_annotations(self):
        video_list = []
        file_list = []
        path_list = []
        with open(self.anno_path, 'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                path = connect_path([self.path, 'frames', self.mode,vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                if num_frames < self.total_length : continue
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                path_list.append(path)
        return video_list, file_list, path_list

    def _get(self, record,image_names,indices,path,idxs):
        images = list()
        for idx in indices:
            if self.generate == "CLIP":
                try:
                    img = self._load_image(record.path, image_names[idx])
                except:
                    print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                    print('invalid indices: {}'.format(indices))
                    raise
            elif self.generate == "Qwen2-VL":
                img = [connect_path([record.path, image_names[idx]])]
            images.extend(img)
        index = np.arange(idxs, idxs + self.total_length, 1, dtype=np.integer)
        return images, record.label, path, index

# HMDB51 data loading class
class HMDB51(ActionDataset):
    def __init__(self, path, act_dict, total_length=12,file_number=1,  random_shift=False, mode='train',generate="CLIP"):
        super().__init__(total_length)
        self.path = path
        self.total_length = total_length
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path,'annotations','hmdb51_' + mode + '_split_' + str(file_number) + '_frames.txt'])
        # self.video_path = connect_path([self.path,'videos'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self.generate = generate
        try:
            self.video_list, self.file_list,self.path_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        path_list = []
        with open(self.anno_path,'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                path = connect_path([self.path,'frames',vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                # path = connect_path([self.video_path, vid]) + '.avi'
                path_list.append(path)
        return video_list, file_list,path_list

    def _get(self, record,image_names,indices,path,idxs):
        images = list()
        for idx in indices:
            if self.generate == "CLIP":
                try:
                    img = self._load_image(record.path, image_names[idx])
                except:
                    print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                    print('invalid indices: {}'.format(indices))
                    raise
            elif self.generate == "Qwen2-VL":
                img = [connect_path([record.path, image_names[idx]])]
            images.extend(img)
        index = np.arange(idxs, idxs + self.total_length, 1, dtype=np.integer)
        return images, record.label,path,index


# ucf101 data loading class
class UCF101(ActionDataset):
    def __init__(self, path, act_dict, total_length=12,file_number=1,  random_shift=False, mode='train',generate="CLIP"):
        super().__init__(total_length)
        self.path = path
        self.total_length = total_length
        self.random_shift = random_shift
        self.mode = mode
        self.anno_path = connect_path([self.path,'annotations','ucf101_' + mode + '_split_' + str(file_number) + '_frames.txt'])
        # self.video_path = connect_path([self.path,'videos'])
        self.act_dict = act_dict
        self.num_classes = len(act_dict)
        self.generate = generate
        try:
            self.video_list, self.file_list,self.path_list = self._parse_annotations()
        except OSError:
            print('ERROR: Could not read annotation file "{}"'.format(self.anno_path))
            raise

    def _parse_annotations(self):
        video_list = []
        file_list = []
        path_list = []
        with open(self.anno_path,'r') as f:
            lines = f.read().splitlines()
            for lin in lines:
                vid, num_frames, labels = lin.split(' ')
                path = connect_path([self.path,'frames',vid])  # 读取RBG
                files = sorted(os.listdir(path))
                num_frames = len(files)
                file_list += [files]
                video_list += [VideoRecord([path, num_frames, int(labels)])]
                # path = connect_path([self.video_path, vid]) + '.avi'
                path_list.append(path)
        return video_list, file_list,path_list

    def _get(self, record,image_names,indices,path,idxs):
        images = list()
        for idx in indices:
            if self.generate == "CLIP":
                try:
                    img = self._load_image(record.path, image_names[idx])
                except:
                    print('ERROR: Could not read image "{}"'.format(connect_path([record.path, image_names[idx]])))
                    print('invalid indices: {}'.format(indices))
                    raise
            elif self.generate == "Qwen2-VL":
                img = [connect_path([record.path, image_names[idx]])]
            images.extend(img)
        index = np.arange(idxs, idxs + self.total_length, 1, dtype=np.integer)
        return images, record.label,path,index