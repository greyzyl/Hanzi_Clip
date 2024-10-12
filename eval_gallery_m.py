import copy
from PIL import Image
import tqdm
import os
import lmdb



os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from model_m import CLIP
import torch.nn as nn
import torchvision.transforms as transforms
import six
from torch.utils.data import Dataset
import json
import zhconv
from tqdm import trange

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

def get_alphabet(alphabet_path):
    
    alp2num = {}
    alphabet_character = []
    alphabet_character.append('PAD')
    lines = open(alphabet_path, 'r').readlines()
    for line in lines:
        alphabet_character.append(line.strip('\n'))
    alphabet_character.append('$')
    for index, char in enumerate(alphabet_character):
        alp2num[char] = index
    return alphabet_character

def get_alphabet_s(alphabet_path_s):
    alp2num_s = {}
    alphabet_character_s = []
    alphabet_character_s.append('PAD')
    lines = list(open(alphabet_path_s, 'r').read())
    for line in lines:
        alphabet_character_s.append(line.strip('\n'))
    alphabet_character_s.append('$')
    for index, char in enumerate(alphabet_character_s):
        alp2num_s[char] = index
    return alphabet_character_s
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
            

            if self.transform is not None:
                img = self.transform(img)
                
        return (img, label[0])
    
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
    
def model_init(vision_type,model_path,radical_alphabet_path,strokelet_alphabet_path,imageW,imageH):
    transform = resizeNormalize((imageW,imageH), test=True)
    alphabet = get_alphabet(radical_alphabet_path)
    alphabet_s = get_alphabet_s(strokelet_alphabet_path)
    model = CLIP(vision_type,embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
             vision_patch_size=32, context_length=50, vocab_size=len(alphabet), stroke_size=len(alphabet_s), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()

    model = nn.DataParallel(model)

    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model,transform

def get_gallery_package(gallery_lmdb_pathes,imageW,imageH):
    train_dataset = []
    for dataset_root in gallery_lmdb_pathes.split(','):
        dataset = lmdbDataset(dataset_root, resizeNormalize((imageW, imageH), test=True))
        train_dataset.append(dataset)
    dataset_total = torch.utils.data.ConcatDataset(train_dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset_total, batch_size=300, shuffle=False, num_workers=8,drop_last=False
    )
    return dataloader

def get_imgfeatures(model,gallery_loader):
    model.eval()
    img_features = []
    
    iters  = len(gallery_loader)
    labels=[]
    gallery_dataloader=iter(gallery_loader)
    with torch.no_grad():
        for i in tqdm.tqdm(range(iters)):
            data = gallery_dataloader.next()
            image, label = data

            labels+=label
            # print(labels)
            # exit()
            imgs = image.cuda()
            img_features_tmp = model.module.encode_image(imgs)
            # img_features_tmp = img_features_tmp / img_features_tmp.norm(dim=1, keepdim=True)
            img_features.append(img_features_tmp)
            # if i>0:
            #     break
    img_features = torch.cat(img_features, dim=0)
    return img_features,labels


def get_alp2num(alphabet_path):
    alp2num = {}
    alphabet_character = []
    alphabet_character.append('PAD')
    lines = open(alphabet_path, 'r').readlines()
    for line in lines:
        alphabet_character.append(line.strip('\n'))
    alphabet_character.append('$')
    for index, char in enumerate(alphabet_character):
        alp2num[char] = index
    return alp2num
def get_alp2num_s(alphabet_path_s):
    ###笔画
    alp2num_s = {}
    alphabet_character_s = []
    alphabet_character_s.append('PAD')
    lines = list(open(alphabet_path_s, 'r').read())
    for line in lines:
        alphabet_character_s.append(line.strip('\n'))
    alphabet_character_s.append('$')
    for index, char in enumerate(alphabet_character_s):
        alp2num_s[char] = index
    return alp2num_s

def get_char_radical_dict(decompose_path):
    dict_file = open(decompose_path, 'r').readlines()
    char_radical_dict = {}
    for line in dict_file:
        line = line.strip('\n')
        try:
            char, r_s = line.split(':')
        except:
            char, r_s = ':', ':'
        if 'rsst' not in decompose_path:
            char_radical_dict[char] = r_s.split(' ')
        else:
            char_radical_dict[char] = list(''.join(r_s.split(' ')))
    return char_radical_dict

def get_char_strokelet_dict(decompose_path_s):
    # 读取字符到笔画序列转换的dict
    dict_file = open(decompose_path_s, 'r').readlines()
    char_stroke_dict = {}
    for line in dict_file:
        line = line.strip('\n')
        try:
            char, r_s = line.split(':')
        except:
            char, r_s = ':', ':'
        if 'rsst' not in decompose_path_s:
            char_stroke_dict[char] = r_s.split(' ')
        else:
            char_stroke_dict[char] = list(''.join(r_s.split(' ')))
    return char_stroke_dict

class Convert_Radical(nn.Module):
    def __init__(self,char_radical_dict,char_strokelet_dict,alp2num,alp2num_s,max_len,max_len_s) -> None:
        super().__init__()
        self.char_radical_dict=char_radical_dict
        self.char_strokelet_dict=char_strokelet_dict
        self.alp2num=alp2num
        self.alp2num_s=alp2num_s
        self.max_len=max_len
        self.max_len_s=max_len_s
    def forward(self,label):
        r_label = []
        s_label = []
        batch = len(label)
        for i in range(batch):
            r_tmp = copy.deepcopy(self.char_radical_dict[label[i]])
            r_tmp.append('$')
            r_label.append(r_tmp)

            s_tmp = copy.deepcopy(self.char_strokelet_dict[label[i]])
            s_tmp.append('$')
            s_label.append(s_tmp)

        text_tensor = torch.zeros(batch, self.max_len).long().cuda()
        stroke_tensor = torch.zeros(batch, self.max_len_s).long().cuda()
        for i in range(batch):
            tmp = r_label[i]
            tmp_s = s_label[i]
            # print(tmp)
            for j in range(len(tmp)):
                text_tensor[i][j] = self.alp2num[tmp[j]]
            for j in range(len(tmp_s)):
                stroke_tensor[i][j] = self.alp2num_s[tmp_s[j]]
        return text_tensor, stroke_tensor



def get_textfeatures(model,texts,convert_radical):
    tmp_text, tmp_text_s = convert_radical(texts)
    text_features = []
    text_features_s=[]
    text_features_all= []
    iters = len(tmp_text) // 100
    with torch.no_grad():
        for i in range(iters+1):
            s = i * 100
            e = (i + 1) * 100
            if e > len(texts):
                e = len(texts)
            if s == len(texts):
                break
            text_features_tmp, seq_text = model.module.encode_text(tmp_text[s:e])
            text_features_tmp_s, seq_text_s = model.module.encode_text_s(tmp_text_s[s:e])
            # print('seq_text:', seq_text.size())
            # print('seq_text_s:', seq_text_s.size())
            text_features_tmp_all = model.module.encode_all(seq_text, seq_text_s)
            text_features.append(text_features_tmp)
            text_features_s.append(text_features_tmp_s)
            text_features_all.append(text_features_tmp_all)
        text_features = torch.cat(text_features, dim=0)
        text_features_s = torch.cat(text_features_s, dim=0)
        text_features_all = torch.cat(text_features_all, dim=0)

    return texts, text_features,text_features_s,text_features_all

def combine_gallery_features(texts,img_features):
    charalphabet_gallery=set(texts)
    char_nums={char:0 for char in charalphabet_gallery}
    char_img_gallery_features={char:torch.zeros([2048]).cuda() for char in charalphabet_gallery}
    for i in range(len(texts)):
        if texts[i] in char_nums.keys():
            char_img_gallery_features[texts[i]]+=img_features[i]
            char_nums[texts[i]]+=1
    for i, (key, value) in enumerate(char_img_gallery_features.items()):
        char_num=char_nums[key]
        char_img_gallery_features[key]/=char_num
        
    img_features=[]
    texts=[]
    for i, (key, value) in enumerate(char_img_gallery_features.items()):
        img_features.append(value.unsqueeze(0))
        texts.append(key)
    img_features=torch.cat(img_features,dim=0)
    
    # print(char_img_gallery_features[texts[200]])
    # print(img_features[200])
    
    return img_features,texts


def build_img_ds(img_root,gt_root):

    json_names=os.listdir(gt_root)
    json_names=[path for path in json_names if path.endswith('.json')]
    img_list=[]
    label_list=[]
    for json_name in json_names:
        with open(os.path.join(gt_root,json_name),'r',encoding='utf-8') as f:
            label=json.load(f)
        for key,value in label.items():
            # if value in char_alpha:
            img_list.append(os.path.join(img_root,json_name[:-5],key))
            label_list.append(value)
    return img_list,label_list

def build_lmdb_ds(lmdb_pathes):
    imgs=[]
    labels=[]
    for lmdb_path in lmdb_pathes:
        env=lmdb.open(
                lmdb_path,
                max_readers=1,
                readonly=True,
                lock=False,
                readahead=False,
                meminit=False)
        with env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            nSamples = nSamples

        with env.begin(write=False) as txn:
            for index in range(1,nSamples+1):
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
                    return 

                label_key = 'label-%09d' % index
                label = str(txn.get(label_key.encode()).decode('utf-8'))
                label = strQ2B(label)
                imgs.append(img)
                labels.append(label)
                
    return imgs,labels


def get_data_package_standard_char(test_datasets_path,batch_size,imageW=128, imageH=128):

    test_dataset = []
    for dataset_root in test_datasets_path:
        dataset = lmdbDataset(dataset_root, resizeNormalize((imageW, imageH), test=True))
        test_dataset.append(dataset)
    test_dataset_total = torch.utils.data.ConcatDataset(test_dataset)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset_total, batch_size=batch_size, shuffle=False, num_workers=8,
    )

    return  test_dataloader
def convert2(label,char_radical_dict,char_stroke_dict,alp2num,alp2num_s):
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

    text_tensor = torch.zeros(batch, 50).long().cuda()
    stroke_tensor = torch.zeros(batch, 50).long().cuda()
    for i in range(batch):
        tmp = r_label[i]
        tmp_s = s_label[i]
        # print(tmp)
        for j in range(len(tmp)):
            text_tensor[i][j] = alp2num[tmp[j]]
        for j in range(len(tmp_s)):
            stroke_tensor[i][j] = alp2num_s[tmp_s[j]]
    return text_tensor, stroke_tensor

def val(model,test_loader,char_radical_dict,char_stroke_dict
        ,alp2num,alp2num_s,char_3755,gallery_lmdb_path,imageH=128,imageW=128):
    print("Start Eval!")

    model.eval()
    test_dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)

    tmp_text, tmp_text_s= convert2(char_3755,char_radical_dict,char_stroke_dict,alp2num,alp2num_s)
    text_features = []
    text_features_s=[]
    text_features_all=[]
    iters = len(tmp_text) // 100
    print('tmp_text',len(tmp_text))
    print('char3755',len(char_3755))
    exit()
    with torch.no_grad():

        #构造img gallery

        gallery_loader=get_gallery_package(gallery_lmdb_path,imageW,imageH)
        img_f,labels=get_imgfeatures(model,gallery_loader)
        standard_char_features,standard_char_gallery=combine_gallery_features(labels,img_f)

        #构造text gallery
        for i in range(iters+1):
            s = i * 100
            e = (i + 1) * 100
            if e > len(char_3755):
                e = len(char_3755)
            if s == len(char_3755):
                break
            text_features_tmp, seq_text = model.module.encode_text(tmp_text[s:e])
            text_features_tmp_s, seq_text_s = model.module.encode_text_s(tmp_text_s[s:e])
            # print('seq_text:', seq_text.size())
            # print('seq_text_s:', seq_text_s.size())
            text_features_tmp_all = model.module.encode_all(seq_text, seq_text_s)
            text_features.append(text_features_tmp)
            text_features_s.append(text_features_tmp_s)
            text_features_all.append(text_features_tmp_all)
        text_features = torch.cat(text_features, dim=0)
        text_features_s = torch.cat(text_features_s, dim=0)
        text_features_all = torch.cat(text_features_all, dim=0)
        correct_r = 0
        correct_i2i = 0
        correct_combine = 0
        correct_s = 0
        correct_all = 0
        total = 0
        #单独获取归一化后的标准字库图像特征
        standard_char_features=standard_char_features/standard_char_features.norm(dim=1, keepdim=True)
        with trange(test_loader_len) as t:
            for iteration in t:
                data = next(test_dataloader)
                image, label = data
                image = torch.nn.functional.interpolate(image, size=(imageH, imageW))

                image = image.cuda()
                text_features_t=text_features.clone()
                image_features, text_features, text_features_s, text_features_all, logit_scale, logit_scale_s, logit_scale_all,logit_scale_i2i = model(image, text_features, text_features_s, text_features_all, test=True)

               
                
                '''
                计算real 2 ids的结果
                '''
                logits_per_image_r = logit_scale.tolist() * image_features @ text_features.t()  # logit_scale.tolist() logit_scale.item() 
                probs_r, index_r = logits_per_image_r.softmax(dim=-1).max(dim=-1)

                logits_per_image_s = logit_scale_s.tolist() * image_features @ text_features_s.t()
                probs_s, index_s = logits_per_image_s.softmax(dim=-1).max(dim=-1)

                logits_per_image_all = logit_scale_all.tolist() * image_features @ text_features_all.t()
                probs_all, index_all = logits_per_image_all.softmax(dim=-1).max(dim=-1)

                '''
                计算real 2 standard char的结果
                '''
                logits_per_image_i2i = logit_scale_i2i.tolist() * image_features @ standard_char_features.t()
                probs_i2i, index_i2i = logits_per_image_i2i.softmax(dim=-1).max(dim=-1)



                for i in range(len(label)):
                    
                    #获取ensemble的预测结果和对应字符序号字典
                    if probs_i2i[i]==max(probs_i2i[i],probs_r[i],probs_s[i],probs_all[i]):#与标准字库匹配的概率最大
                        index_combine = index_i2i[i]
                        char_dict_combine=standard_char_gallery
                    else:#与ids匹配的概率最大
                        if probs_all[i]==max(probs_i2i[i],probs_r[i],probs_s[i],probs_all[i]):
                            index_combine = index_all[i]
                        elif probs_s[i]==max(probs_i2i[i],probs_r[i],probs_s[i],probs_all[i]):
                            index_combine = index_s[i]
                        else:
                            index_combine = index_r[i]
                        char_dict_combine=char_3755
                    
                    # print('pred',char_3755[index_r[i]])
                    # print('gt',label[i])
                    #各自判断正确率
                    if char_3755[index_r[i]] == label[i]:
                        correct_r += 1
                    if char_3755[index_s[i]] == label[i]:
                        correct_s += 1
                    if char_3755[index_all[i]] == label[i]:
                        correct_all += 1
                    if standard_char_gallery[index_i2i[i]] == label[i]:
                        correct_i2i += 1
                    if char_dict_combine[index_combine] == label[i]:
                        correct_combine += 1
                    total += 1
                t.set_description('{}/{}'.format(iteration, test_loader_len))
                t.set_postfix(acc=(max([correct_i2i,correct_r,correct_s,correct_all,correct_combine]) / total))
        print(total)
        print("ACC : {}".format(max([correct_r,correct_s,correct_all,correct_i2i,correct_combine]) / total))
        print("ACC_r : {}".format(correct_r / total))
        print("ACC_s : {}".format(correct_s / total))
        print("ACC_ids_all : {}".format(correct_all / total))
        print("ACC_i2i : {}".format(correct_i2i / total))
        print("ACC_combine : {}".format(correct_combine / total))


if __name__=='__main__':
    #load 测试数据超参
    imageH,imageW=128,128
    # test_gt_root='/mnt/disk1/luwei/Archives_char_recognition/img_crops/crop_gts'
    # test_img_root='/mnt/disk1/luwei/Archives_char_recognition/img_crops/crop_imgs'
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/SIKU_val_v1','/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/v1_test']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/guji']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/HWDB1.0test']
    test_lmdb_root=['/home/yuhaiyang/dataset/CharacterZeroShot/test_1000']

    #load 部首gallery超参
    decompose_path='./data/decompose.txt'
    decompose_path_s='./data/27533_rsst_decompose.txt'
    # radical_alphabet_path='data/radical_all_Chinese.txt'
    radical_alphabet_path='./data/radical_3755.txt'
    strokelet_alphabet_path='./data/rsst_alphabet_r.txt'
    radical_max_len=50
    strokelet_max_len=50
    radical_char_alpha_path = './data/char_3755.txt'

    #load 图像gallery超参
    # gallery_lmdb_path='/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char/byte_data'
    gallery_lmdb_path='/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char/char_3755'
    # gallery_char_dict='/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/statistic_char_in_dataset_char_list.json'
    
    #load 模型超参并初始化模型及图像转化器
    ckpt_path='history/【10-09】hwdb500_仅图文_data_sampler_resnet_no_gard/best_model_all.pth'
    vision_type='resnet'
    # ckpt_path='/home/yuhaiyang/CLIP-OCR/history/【09-23】vit_clipgrad_loss_combine_bytedata/best_model.pth'
    # vision_type='vit'
    model,transform=model_init(vision_type,ckpt_path,radical_alphabet_path,strokelet_alphabet_path,imageW,imageH)
    test_loader=build_lmdb_ds(test_lmdb_root)




    #build radical gallery
    char_file = open(radical_char_alpha_path, 'r').read()
    char_alpha = list(char_file)
    char_radical_dict=get_char_radical_dict(decompose_path)
    char_strokelet_dict=get_char_strokelet_dict(decompose_path_s)
    alp2num=get_alp2num(radical_alphabet_path)
    alp2num_s=get_alp2num_s(strokelet_alphabet_path)
    convert_radical=Convert_Radical(char_radical_dict,char_strokelet_dict,alp2num,alp2num_s
                                    ,radical_max_len,strokelet_max_len)
    char_alpha, text_features,text_features_s,text_features_all = get_textfeatures(model,char_alpha,convert_radical)

    test_loader=get_data_package_standard_char(test_lmdb_root,batch_size=300)
    val(model,test_loader,char_radical_dict,char_strokelet_dict,
        alp2num,alp2num_s,char_alpha,gallery_lmdb_path)

