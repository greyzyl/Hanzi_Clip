import copy
from PIL import Image
import tqdm
import os
import lmdb


os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import torch
from model import CLIP
import torch.nn as nn
import torchvision.transforms as transforms
import six
from torch.utils.data import Dataset
import json
import zhconv

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
    
def model_init(vision_type,model_path,radical_alphabet_path,imageW,imageH):
    transform = resizeNormalize((imageW,imageH), test=True)
    alphabet = get_alphabet(radical_alphabet_path)
    model = CLIP(vision_type=vision_type,embed_dim=2048, context_length=30, 
            vocab_size=len(alphabet), transformer_width=512,
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

class Convert_Radical(nn.Module):
    def __init__(self,char_radical_dict,alp2num,max_len) -> None:
        super().__init__()
        self.char_radical_dict=char_radical_dict
        self.alp2num=alp2num
        self.max_len=max_len
    def forward(self,label):
        r_label = []
        batch = len(label)
        for i in range(batch):
            
            r_tmp = copy.deepcopy(self.char_radical_dict[label[i]])
            r_tmp.append('$')
            r_label.append(r_tmp)

        # print(r_label)
        text_tensor = torch.zeros(batch, self.max_len).long().cuda()
        for i in range(batch):
            tmp = r_label[i]
            
            for j in range(len(tmp)):
                try:
                    text_tensor[i][j] = self.alp2num[tmp[j]]
                except:
                    pass
        return text_tensor



def get_textfeatures(model,texts,convert_radical):
    tmp_text = convert_radical(texts)
    text_features = []
    iters = len(tmp_text) // 100
    with torch.no_grad():
        for i in range(iters+1):
            s = i * 100
            e = (i + 1) * 100
            if e > len(tmp_text):
                e = len(tmp_text)
            text_features_tmp = model.module.encode_text(tmp_text[s:e])
            text_features.append(text_features_tmp)
        text_features = torch.cat(text_features, dim=0)

    return texts, text_features

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

def inference_one_img(image, text_features, texts,imageH,imageW,transform,label_idx=None, transformed=False):
    model.eval()

    if not transformed:
        image = transform(image)

    image = image.unsqueeze(dim=0)
    image = torch.nn.functional.interpolate(image, size=(imageH,imageW))

    image = image.cuda()
    
    image_features, text_features, logit_scale = model(image, text_features, test=True)
    logits_per_image = logit_scale.item() * image_features @ text_features.t()
    probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
    res = texts[index[0]]
    # if label_idx!=None:
    #     print(logits_per_image.softmax(dim=-1)[:,label_idx])
    #     print(probs[0])
    return res,probs[0]

def inference_one_img_gallery(image, gallery_features, texts,imageH,imageW,transform, transformed=False):
    model.eval()

    if not transformed:
        image = transform(image)

    image = image.unsqueeze(dim=0)
    image = torch.nn.functional.interpolate(image, size=(imageH, imageW))

    image = image.cuda()
    
    image_features, gallery_features, logit_scale = model(image, gallery_features, test=True)
    logits_per_image = logit_scale.item() * image_features @ gallery_features.t()
    
    probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)

    res = texts[index[0]]

    return res,probs[0]

if __name__=='__main__':
    #load 测试数据超参
    imageH,imageW=128,128
    # test_gt_root='/mnt/disk1/luwei/Archives_char_recognition/img_crops/crop_gts'
    # test_img_root='/mnt/disk1/luwei/Archives_char_recognition/img_crops/crop_imgs'
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/SIKU_val_v1','/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/v1_test']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/test/guji']
    # test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/HWDB1.0test']
    test_lmdb_root=['/home/yuhaiyang/dataset/hanziclip_data_lmdb/danganguan/v1_test']
    #load 部首gallery超参
    decompose_path='./data/decompose.txt'
    # radical_alphabet_path='data/radical_all_Chinese.txt'
    radical_alphabet_path='/home/yuhaiyang/CLIP-OCR/data/radical_27533.txt'
    radical_max_len=30
    radical_char_alpha_path = './data/all_chinese.txt'

    #load 图像gallery超参
    # gallery_lmdb_path='/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char/byte_data'
    gallery_lmdb_path='/home/yuhaiyang/dataset/hanziclip_data_lmdb/standard_char/char_27533'
    # gallery_char_dict='/home/yuhaiyang/dataset/hanziclip_data_lmdb/byte_guji_data/statistic_char_in_dataset_char_list.json'
    
    #load 模型超参并初始化模型及图像转化器
    ckpt_path='history/【09-22】combine_loss_pretrain/best_model.pth'
    vision_type='vit'
    # ckpt_path='/home/yuhaiyang/CLIP-OCR/history/【09-23】vit_clipgrad_loss_combine_bytedata/best_model.pth'
    # vision_type='vit'
    model,transform=model_init(vision_type,ckpt_path,radical_alphabet_path,imageW,imageH)

    #build radical gallery
    char_file = open(radical_char_alpha_path, 'r').read()
    char_alpha = list(char_file)
    char_radical_dict=get_char_radical_dict(decompose_path)
    alp2num=get_alp2num(radical_alphabet_path)
    convert_radical=Convert_Radical(char_radical_dict,alp2num,radical_max_len)
    char_alpha, text_features = get_textfeatures(model,char_alpha,convert_radical)


    #build img gallery
    gallery_loader=get_gallery_package(gallery_lmdb_path,imageW,imageH)
    img_f,labels=get_imgfeatures(model,gallery_loader)
    img_features,texts_gallery=combine_gallery_features(labels,img_f)


    #infer or eval
    # img_list,label_list=build_img_ds(test_img_root,test_gt_root)#也可以直接给图片路径的list
    img_list,label_list=build_lmdb_ds(test_lmdb_root)
    correct, total = 0, 0
    t_win,i_win=0,0
    pre_record=[]
    for i in tqdm.tqdm(range(len(img_list))):
        img, label = img_list[i],label_list[i]
        if type(img)==str:
            img=Image.open(img)
        try:
        #只使用text encoder的编码作为gallery
            pred,_ = inference_one_img(img, text_features, char_alpha,imageH=imageH,imageW=imageW,label_idx=char_alpha.index(label),transform=transform, transformed=False)

            #只使用image encoder的编码作为gallery
            # pred,probs = inference_one_img_gallery(img, img_features, texts_gallery,imageH=imageH,imageW=imageW,transform=transform, transformed=False)
            
            # pred_t,prob_t = inference_one_img(img, text_features, char_alpha,imageH=imageH,imageW=imageW,transform=transform, transformed=False)
            # pred_i,prob_i = inference_one_img_gallery(img, img_features, texts_gallery,imageH=imageH,imageW=imageW,transform=transform, transformed=False)
            # # print(prob_t)
            # if pred_t == label and pred_i!=label:
            #     t_win+=1
            # elif pred_i == label and pred_t!=label:
            #     i_win+=1
            # # print(prob_i)
            # if prob_i>prob_t:
            #     pred=pred_i
            # else:
            #     pred=pred_t

            #根据label选择使用gallery
            # if label in char_alpha:
            #     pred_t,prob_t = inference_one_img(img, text_features, char_alpha,imageH=imageH,imageW=imageW,transform=transform, transformed=False)
            #     pred_i,prob_i = inference_one_img_gallery(img, img_features, texts_gallery,imageH=imageH,imageW=imageW,transform=transform, transformed=False)
            #     # print(prob_t)
            #     if pred_t == label and pred_i!=label:
            #         t_win+=1
            #     elif pred_i == label and pred_t!=label:
            #         i_win+=1
            #     # print(prob_i)
            #     if prob_i>prob_t:

            #         pred=pred_i
            #     else:
            #         pred=pred_t
            # else:
            #     pred,prob = inference_one_img_gallery(img, img_features, texts_gallery,imageH=imageH,imageW=imageW,transform=transform, transformed=False)

            if traditional_to_simplify(pred) == traditional_to_simplify(label):
                correct += 1
            else:

                img.save(f'workdir/gt_{traditional_to_simplify(label)}_pred_{traditional_to_simplify(pred)}.png')
            # else:
            #     print(pred,label)
            total += 1
        except:
            print(f'error{label}')
        pre_record.append({'idx':i,'pre':pred,'label':label})
    record={
        'data_pathes':test_lmdb_root,
        'record':pre_record
    }
    with open(os.path.join(os.path.dirname(ckpt_path),'eval_record.txt'),'w',encoding='utf=8') as f:
        json.dump(record,f,ensure_ascii=False,indent=4)
    print('t_win',t_win)
    print('i_win',i_win)
    print(f'ACC: {correct/total}, correct: {correct}, total: {total}')
