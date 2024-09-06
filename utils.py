import torch
import time
import shutil
import os

from config import config
from dataset import lmdbDataset, resizeNormalize
import torch
import shutil
from shutil import copyfile
import os
import torch.nn.functional as F
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

# 读取字符到部首序列转换的dict
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
    # for dataset_root in config['train_dataset'].split(','):
    for dataset_root in config['train_dataset']:
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=False))
        train_dataset.append(dataset)
    train_dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset_total, batch_size=config['batch'], shuffle=True, num_workers=8,
    )

    test_dataset = []
    # for dataset_root in config['test_dataset'].split(','):
    for dataset_root in config['test_dataset']:
        dataset = lmdbDataset(dataset_root, resizeNormalize((config['imageW'], config['imageH']), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=config['batch'], shuffle=False, num_workers=8,
    )

    return train_dataloader, test_dataloader

def pad_and_combine(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    将具有 batch 维度的张量 A 和 B 通过 padding 组合成一个新的张量 C.
    
    填充值为 -inf。

    参数:
    A (torch.Tensor): 形状为 (batch_size, height_A, width_A) 的张量
    B (torch.Tensor): 形状为 (batch_size, height_B, width_B) 的张量
    
    返回:
    torch.Tensor: 组合后的张量 C，形状为 (batch_size, max(height_A + height_B), max(width_A + width_B))
    """
    # 确保 A 和 B 有相同的 batch_size
    assert A.size(0) == B.size(0), "A 和 B 必须有相同的 batch 大小"
    
    batch_size = A.size(0)
    
    # 计算每个 batch 中 A 和 B 的高度和宽度
    height_A, width_A = A.size(1), A.size(2)
    height_B, width_B = B.size(1), B.size(2)
    
    # 计算需要填充的尺寸
    A_padding_right = width_B  # A 的右侧填充
    A_padding_bottom = height_B  # A 的下方填充
    B_padding_top = height_A  # B 的上方填充
    B_padding_left = width_A  # B 的左侧填充

    # 给 A 加 padding，填充值为 -inf
    A_padded = F.pad(A, (0, A_padding_right, 0, A_padding_bottom), value=-float('inf'))

    # 给 B 加 padding，填充值为 -inf
    B_padded = F.pad(B, (B_padding_left, 0, B_padding_top, 0), value=-float('inf'))

    # 将 A 和 B 相加组合成 C
    C = torch.max(A_padded, B_padded)  # 使用 max 以确保组合时不受 -inf 影响

    return C

import copy
def convert(label):
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

    text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    text_tensor_mask = torch.empty(batch,config['max_len'], config['max_len'])
    text_tensor_mask.fill_(float("-inf"))
    text_tensor_mask.triu_(1)
    stroke_tensor = torch.zeros(batch, config['max_len_s']).long().cuda()
    stroke_tensor_mask = torch.empty(batch,config['max_len_s'], config['max_len_s'])
    stroke_tensor_mask.fill_(float("-inf"))
    stroke_tensor_mask.triu_(1)
    fused_tensor_mask= torch.empty(batch,config['max_len']+config['max_len_s'], config['max_len']+config['max_len_s'])
    fused_tensor_mask.fill_(float("-inf"))
    text_2_stroke_cross_attn_mask=torch.empty(batch,config['max_len'], config['max_len_s'])
    text_2_stroke_cross_attn_mask.fill_(float("-inf"))
    stroke_2_text_cross_attn_mask=torch.empty(batch,config['max_len_s'], config['max_len'])
    stroke_2_text_cross_attn_mask.fill_(float("-inf"))
    for i in range(batch):
        tmp = r_label[i]
        tmp_s = s_label[i]
        # print(tmp_s)
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
        for j in range(len(tmp_s)):
            stroke_tensor[i][j] = alp2num_s[tmp_s[j]]
        text_tensor_mask[i,:,len(tmp):]=float("-inf")
        stroke_tensor_mask[i,:,len(tmp_s):]=float("-inf")
        fused_tensor_mask[i,:,0:len(tmp)]=0
        fused_tensor_mask[i,:,config['max_len']:config['max_len']+len(tmp_s)]=0
        text_2_stroke_cross_attn_mask[i,:,:len(tmp_s)]=0
        stroke_2_text_cross_attn_mask[i,:,:len(tmp)]=0
    # fused_tensor_mask=pad_and_combine(text_tensor_mask,stroke_tensor_mask)
    # fused_tensor_mask[:,len(tmp):,0:len(tmp)]=0

    
    return text_tensor, stroke_tensor,text_tensor_mask,stroke_tensor_mask,fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask


def convert2(label):
    r_label = []
    s_label = []
    batch = len(label)
    for i in range(batch):
        r_tmp = copy.deepcopy(char_radical_dict[label[i]])
        r_tmp.append('$')
        r_label.append(len(r_tmp))

        s_tmp = copy.deepcopy(char_stroke_dict[label[i]])
        s_tmp.append('$')
        s_label.append(len(s_tmp))

    # text_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    # stroke_tensor = torch.zeros(batch, config['max_len']).long().cuda()
    # for i in range(batch):
    #     tmp = r_label[i]
    #     tmp_s = s_label[i]
    #     # print(tmp)
    #     for j in range(len(tmp)):
    #         text_tensor[i][j] = alp2num[tmp[j]]
    #     for j in range(len(tmp_s)):
    #         stroke_tensor[i][j] = alp2num_s[tmp_s[j]]
    return r_label, s_label
def saver():
    try:
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
if __name__=='__main__':
    text_tensor, stroke_tensor,text_tensor_mask,stroke_tensor_mask,fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask=convert(['我'])
    # print(text_tensor_mask[0])
    # print(stroke_tensor_mask[0])
    # print(fused_tensor_mask[0])
    print(text_2_stroke_cross_attn_mask[0])
    print(stroke_2_text_cross_attn_mask[0])
    # print(text_tensor_mask[1])
    # print(stroke_tensor_mask[1])
    # print(fused_tensor_mask[1])
