import random
import numpy as np
from config import config
from dataset import ConcatDatasetWithGetLabel, lmdbDataset, lmdbDataset_sampler, lmdbDataset_standard_char_sampler, resizeNormalize,lmdbDataset_standard_char
import torch
import shutil
from shutil import copyfile
import os
from collections import defaultdict
from torch.utils.data import Dataset
from torch.utils.data import Sampler, DataLoader
# 读取字符集合
alp2num = {}
alphabet_character = []
alphabet_character.append('PAD')
lines = open(config['alphabet_path'], 'r').readlines()
for line in lines:
    alphabet_character.append(line.strip('\n'))
alphabet_character.append('$')
for index, char in enumerate(alphabet_character):
    alp2num[char] = index

###笔画
alp2num_s = {}
alphabet_character_s = []
alphabet_character_s.append('PAD')
lines = list(open(config['alphabet_path_s'], 'r').read())
for line in lines:
    alphabet_character_s.append(line.strip('\n'))
alphabet_character_s.append('$')
for index, char in enumerate(alphabet_character_s):
    alp2num_s[char] = index

#读取字符到部首序列转换的dict
dict_file = open(config['decompose_path'], 'r').readlines()
char_radical_dict = {}
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    if 'rsst' not in config['decompose_path']:
        char_radical_dict[char] = r_s.split(' ')
    else:
        char_radical_dict[char] = list(''.join(r_s.split(' ')))

# 读取字符到笔画序列转换的dict
dict_file = open(config['decompose_path_s'], 'r').readlines()
char_stroke_dict = {}
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    if 'rsst' not in config['decompose_path_s']:
        char_stroke_dict[char] = r_s.split(' ')
    else:
        char_stroke_dict[char] = list(''.join(r_s.split(' ')))

def get_data_package():
    train_dataset = []
    for dataset_root in config['train_dataset']:
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=8,
    )

    test_dataset = []
    for dataset_root in config['test_dataset']:
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=8,
    )

    return train_dataloader, test_dataloader

def get_train_dataset_sampler():
    train_dataset = []
    for dataset_root in config['train_dataset']:
        dataset = lmdbDataset_sampler(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = ConcatDatasetWithGetLabel(train_dataset)
    return train_dataset_total

def get_data_package_standard_char():
    train_dataset = []
    for dataset_root in config['train_dataset']:
        dataset = lmdbDataset_standard_char(dataset_root,config['standard_char_dataset'], resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=8,
    )

    test_dataset = []
    for dataset_root in config['test_dataset']:
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=8,
    )

    return train_dataloader, test_dataloader
def get_train_dataset_standard_char_sampler():
    train_dataset = []
    for dataset_root in config['train_dataset']:
        dataset = lmdbDataset_standard_char_sampler(dataset_root,config['standard_char_dataset'], resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = ConcatDatasetWithGetLabel(train_dataset)
    return train_dataset_total

import copy
def convert(label):
    r_label = []
    s_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        # print(tmp)
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
    return text_tensor
def convert2(label):
    r_label = []
    s_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(r_tmp)

        s_tmp = copy.deepcopy(char_stroke_dict[label[i]])
        s_tmp.append('$')
        s_label.append(s_tmp)
        # print(s_tmp)
        # print(alp2num_s)
        # exit()

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    stroke_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        tmp_s = s_label[i]
        # print(tmp)
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
        for j in range(len(tmp_s)):
            stroke_tensor[i][j] = alp2num_s[tmp_s[j]]
    return text_tensor, stroke_tensor

class RandomIdentitySampler(Sampler):
    """
    Randomly sample N identities, then for each identity,
    randomly sample K instances, therefore batch size is N*K.
    Args:
    - data_source (list): list of (img_path, pid, camid).
    - num_instances (int): number of instances per identity in a batch.
    - batch_size (int): number of examples in a batch.
    """

    def __init__(self, data_source, batch_size, num_instances):
        self.data_source = data_source
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        self.index_dic = defaultdict(list)
        # print('start loading label')
        # start=time.time()
        # for index, ( _,pid) in enumerate(self.data_source):
        #     if index==len(data_source)-1:
        #         break
        #     self.index_dic[pid].append(index)
        for index in range(len(self.data_source)):
            pid=self.data_source.get_label(index)
            self.index_dic[pid].append(index)
        self.pids = list(self.index_dic.keys())
        # end=time.time()
        # with open('time.txt',"w") as f:
        #     f.write(f'loading label cost:{end-start}\n')
        # estimate number of examples in an epoch
        self.length = 0
        # print('start counting char num')
        # start=time.time()
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances
        # end=time.time()
        # with open('time.txt',"a") as f:
        #     f.write(f'counting char num cost:{end-start}\n')        
    def __iter__(self):
        batch_idxs_dict = defaultdict(list)
        # print('start first sampling')
        # start=time.time()
        for pid in self.pids:
            idxs = copy.deepcopy(self.index_dic[pid])
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            random.shuffle(idxs)
            batch_idxs = []
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []
        # end=time.time()
        # print(f'first sampling cost:{end-start}')
        # with open('time.txt',"a") as f:
        #     f.write(f'first sampling cost:{end-start}\n') 
        avai_pids = copy.deepcopy(self.pids)
        final_idxs = []
        # print('start second sampling')
        # start=time.time()
        while len(avai_pids) >= self.num_pids_per_batch:
            selected_pids = random.sample(avai_pids, self.num_pids_per_batch)
            for pid in selected_pids:
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)
        # end=time.time()
        # print(f'second sampling:{end-start}s')
        # with open('time.txt',"a") as f:
        #     f.write(f'second sampling:{end-start}s\n') 
        self.length = len(final_idxs)
        return iter(final_idxs)

    def __len__(self):
        return self.length
def saver():
    try:
        # 将原有的文件删掉
        shutil.rmtree('./history/{}'.format(config['exp_name']))
    except:
        pass
    os.mkdir('./history/{}'.format(config['exp_name']))

    # 还可以打个时间戳
    import time

    print('**** Experiment Name: {} ****'.format(config['exp_name']))

    localtime = time.asctime(time.localtime(time.time()))
    f = open(os.path.join('./history', config['exp_name'], str(localtime)),'w+')
    f.close()

    # 开始复制文件
    src = './main_m.py'
    dst = os.path.join('./history', config['exp_name'], 'main_m.py')
    copyfile(src, dst)

    src = './utils.py'
    dst = os.path.join('./history', config['exp_name'], 'utils.py')
    copyfile(src, dst)

    src = './dataset.py'
    dst = os.path.join('./history', config['exp_name'], 'dataset.py')
    copyfile(src, dst)

    src = './clip.py'
    dst = os.path.join('./history', config['exp_name'], 'clip.py')
    copyfile(src, dst)

    src = './model_m.py'
    dst = os.path.join('./history', config['exp_name'], 'model_m.py')
    copyfile(src, dst)

    src = './config.py'
    dst = os.path.join('./history', config['exp_name'], 'config.py')
    copyfile(src, dst)

    src = './resnet50.py'
    dst = os.path.join('./history', config['exp_name'], 'resnet50.py')
    copyfile(src, dst)

    src = './resnet.py'
    dst = os.path.join('./history', config['exp_name'], 'resnet.py')
    copyfile(src, dst)

def get_alphabet():
    return alphabet_character
def get_alphabet_s():
    return alphabet_character_s
