import lmdb
import sys
import six
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from PIL import Image
from torch.utils.data import Dataset

def read_all_Chinese():
    path = './data/all_chinese.txt'
    with open(path, 'r') as f:
        data = f.read()
    data = set(list(data))
    return data

all_Chinese = read_all_Chinese()

# img_path = '/home/luwei/code/image-ids-CTR/微信图片_20240603182408.png'
# img_temp = Image.open(img_path).convert('RGB')
# label_temp = '工'

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
        assert index <= len(self), 'index range error index: %d' % index
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
            
            # print(label)
            if label not in all_Chinese:
                img = img_temp
                label = label_temp

            if self.transform is not None:
                img = self.transform(img)
                
        return (img, label[0])

def strQ2B(ustring):
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248

        rstring += chr(inside_code)
    return rstring

class resizeNormalize(object):

    def __init__(self, size, interpolation=Image.BILINEAR, test=False):
        self.size = size
        self.test = test
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


if __name__ == '__main__':
    ds_path = '/mnt/disk1/luwei/Archives_char_recognition/dataset/archives_test2'
    ds = lmdbDataset(ds_path)
    print(len(ds))