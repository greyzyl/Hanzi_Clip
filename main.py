import os

import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import Dataset
from tqdm import trange
from config import config
from dataset import lmdbDataset, resizeNormalize
from utils import get_data_package, convert, saver, get_alphabet
from model import CLIP
from torch.utils.data import Sampler, DataLoader
import zhconv
import lmdb
import copy
import random
import six
from PIL import Image
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
class lmdbDataset_my(Dataset):

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
            # img_key = 'image-%09d' % index
            # imgbuf = txn.get(img_key.encode())

            # buf = six.BytesIO()
            # buf.write(imgbuf)
            # buf.seek(0)
            # try:
            #     img = Image.open(buf).convert('RGB')
            #     pass
            # except IOError:
            #     print('Corrupted image for %d' % index)
            #     return self[index + 1]

            label_key = 'label-%09d' % index
            label = str(txn.get(label_key.encode()).decode('utf-8'))
            label = strQ2B(label)
            
                
        return label[0]
    def __getitem__(self, index):
        # if index > len(self):
        #     index = len(self) - 1
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
            

            if self.transform is not None:
                img = self.transform(img)
                
        return (img, label[0])
        
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

def simplify_to_traditional(text):
    """
    将简体字转换为繁体字
    :param text: str, 需要转换的简体字文本
    :return: str, 转换后的繁体字文本
    """
    return zhconv.convert(text, 'zh-tw')

def traditional_to_simplify(text):
    """
    将繁体字转换为简体字
    :param text: str, 需要转换的繁体字文本
    :return: str, 转换后的简体字文本
    """
    return zhconv.convert(text, 'zh-cn')


alphabet = get_alphabet()
# model = CLIP(embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
#              vision_patch_size=32, context_length=30, vocab_size=len(alphabet), transformer_width=512,
#              transformer_heads=8, transformer_layers=12).cuda()
model = CLIP('vit',embed_dim=2048, context_length=30, 
            vocab_size=len(alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
model = nn.DataParallel(model)


if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))
    print('loading！！！')


params_grad = [p.numel() for n, p in model.named_parameters()]
print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

_, test_loader = get_data_package()

train_dataset = lmdbDataset_my(config['train_dataset'][0], resizeNormalize((config['imageW'], config['imageH']), test=False))

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-6)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

char_file = open('./data/char_1000_test.txt', 'r').read()
char_3755 = list(char_file)


global best_acc
best_acc = -1

def val(model):
    print("Start Eval!")

    model.eval()
    test_dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

    tmp_text = convert(char_3755)
    text_features = []
    iters = len(tmp_text) // 100
    with torch.no_grad():
        for i in range(iters+1):
            s = i * 100
            e = (i + 1) * 100
            if e > len(char_3755):
                e = len(char_3755)
            text_features_tmp = model.module.encode_text(tmp_text[s:e])
            text_features.append(text_features_tmp)
        text_features = torch.cat(text_features, dim=0)
        correct = 0
        total = 0
        with trange(test_loader_len) as t:
            for iteration in t:
                data = next(test_dataloader)
                image, label = data
                image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

                image = image.cuda()
                image_features, text_features, logit_scale = model(image, text_features, test=True)
                logits_per_image = logit_scale.tolist()[0] * image_features @ text_features.t()  # logit_scale.tolist()[0] logit_scale.item() 
                probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
                for i in range(len(label)):
                    if char_3755[index[i]] == label[i]:
                        correct += 1
                    total += 1
                t.set_description('{}/{}'.format(iteration, test_loader_len))
                t.set_postfix(acc=correct/total)
        print(total)
        print("ACC : {}".format(correct / total))
        global best_acc
        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), './history/{}/best_model.pth'.format(config['exp_name']))
    f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {}\n".format(epoch, correct / total))
    f.close()

saver()

if config['test_only']:
    val(model)
    exit()


for epoch in range(config['epoch']):
    sampler = RandomIdentitySampler(train_dataset, batch_size=config['batch'],num_instances=1)
    train_loader = DataLoader(train_dataset,batch_size=config['batch'], sampler=sampler, num_workers=2,
                pin_memory=True, drop_last=True,)
    dataloader = iter(train_loader)
    train_loader_len = len(train_loader)
    print('training:', train_loader_len)
    for iteration in range(train_loader_len):
        model.train()
        optimizer.zero_grad()

        data = next(dataloader)
        image, label = data

        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
        text = convert(label)

        image = image.cuda()
        text = text.cuda()
        image_features, text_features, logit_scale = model(image, text)
        logits_per_image = logit_scale.tolist()[0] * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        label_str = ''.join(label)
        ground_truth = torch.arange(len(image), dtype=torch.long).cuda()
        for i in range(len(image)):
            ground_truth[i] = label_str.index(label[i])
        total_loss = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2

        total_loss.backward()
        optimizer.step()

        print('epoch:{}, iter:{}/{}, loss:{}'.format(epoch, iteration, train_loader_len, total_loss))
        

    val(model)
    if (epoch + 1) > 10 and (epoch + 1) % 2 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.8
