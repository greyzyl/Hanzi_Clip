import random
import torch
from torch.utils.data import Dataset,ConcatDataset
from torch.utils.data import sampler
import torchvision.transforms as transforms
import lmdb
import sys
import cv2
from PIL import Image
import numpy as np
import six
from PIL import Image
from data.augment import distort, stretch, perspective
import config
def read_all_Chinese():
    path = config['char_alphabet_path']
    with open(path, 'r') as f:
        data = f.read()
    data = set(list(data))
    return data
all_Chinese = read_all_Chinese()

img_path = 'data/微信图片_20240603182408.png'
img_temp = Image.open(img_path).convert('RGB')
label_temp = '工'
def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

class lmdbDataset(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            if label not in all_Chinese:
                img = img_temp
                label = label_temp
            if self.transform is not None:
                img = self.transform(img)
        return (img, label)

class lmdbDataset_sampler(Dataset):

    def __init__(self, root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def get_label(self,index):
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)        
        return label[0]
    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)

            if self.transform is not None:
                img = self.transform(img)
        return (img, label)


class lmdbDataset_standard_char(Dataset):

    def __init__(self, root=None,standard_char_root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        #构建标准字库图像字典
        self.transform = transform
        self.stardard_char =lmdb.open(
            standard_char_root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        with self.stardard_char.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples_standard_char = nSamples
        
        self.standard_char_dict = {}
        with self.stardard_char.begin(write=False) as txn:
            for i in range(1,self.nSamples_standard_char+1):
                standard_char_key = 'label-%09d' % i
                standard_char = str(txn.get(standard_char_key.encode()).decode('utf-8'))
                standard_char = strQ2B(standard_char)
                standard_char_img_key = 'image-%09d' % i
                imgbuf = txn.get(standard_char_img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('RGB')
                    pass
                except IOError:
                    print('Corrupted image for %d' % i)
                    return self[i + 1]
                if self.transform is not None:
                    img = self.transform(img)
                self.standard_char_dict[standard_char] = img.clone()




        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            standard_char_img = self.standard_char_dict[label]
            if self.transform is not None:
                img = self.transform(img)
        return (img, label,standard_char_img)

class lmdbDataset_standard_char_sampler(Dataset):

    def __init__(self, root=None,standard_char_root=None, transform=None, reverse=False, alphabet=None):
        self.env = lmdb.open(
            root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        #构建标准字库图像字典
        self.transform = transform
        self.stardard_char =lmdb.open(
            standard_char_root,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False)
        
        with self.stardard_char.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples_standard_char = nSamples
        
        self.standard_char_dict = {}
        with self.stardard_char.begin(write=False) as txn:
            for i in range(1,self.nSamples_standard_char+1):
                standard_char_key = 'label-%09d' % i
                standard_char = str(txn.get(standard_char_key.encode()).decode('utf-8'))
                standard_char = strQ2B(standard_char)
                standard_char_img_key = 'image-%09d' % i
                imgbuf = txn.get(standard_char_img_key.encode())
                buf = six.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                try:
                    img = Image.open(buf).convert('RGB')
                    pass
                except IOError:
                    print('Corrupted image for %d' % i)
                    return self[i + 1]
                if self.transform is not None:
                    img = self.transform(img)
                self.standard_char_dict[standard_char] = img.clone()




        if not self.env:
            print('cannot creat lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

        self.transform = transform
        self.reverse = reverse

    def __len__(self):
        return self.nSamples
    def get_label(self,index):
        index += 1
        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)        
        return label[0]
    
    def __getitem__(self, index):
        if index > len(self):
            index = len(self) - 1
        assert index <= len(self), 'index range error 报错index为 %d' % index
        index += 1
        with self.env.begin(write=False) as txn:
            img_key = 'image-%09d' % index
            imgbuf = txn.get(img_key.encode())

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                img = Image.open(buf).convert('RGB')
                pass
            except IOError:
                print('Corrupted image for %d' % index)
                return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            standard_char_img = self.standard_char_dict[label]
            if self.transform is not None:
                img = self.transform(img)
        return (img, label,standard_char_img)


class ConcatDatasetWithGetLabel(ConcatDataset):
    def get_label(self, index):
        # 找到对应的子数据集及其索引
        for dataset in self.datasets:
            if index < len(dataset):
                return dataset.get_label(index)
            index -= len(dataset)
        raise IndexError("Index out of range")

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, test=False):
        self.size = size
        self.test = test
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)

        # if not self.test:
        #     img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        #     # 数据增强
        #     aug = random.randint(0, 1)
        #     if aug == 1:
        #         img = distort(img, 4)
        #
        #     aug_1 = random.randint(0, 1)
        #     if aug_1 == 1:
        #         img = stretch(img, 4)
        #
        #     aug_2 = random.randint(0, 1)
        #     if aug_2 == 1:
        #         img = perspective(img)
        #
        #     img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img
if __name__ == '__main__':
    dataset=lmdbDataset_standard_char('/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/train/patch_SIKU','/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char',resizeNormalize((128,128), test=False))
    img=dataset.standard_char_dict['我']
    img=img.permute(1,2,0).numpy()*255
    cv2.imwrite('t.png',img)