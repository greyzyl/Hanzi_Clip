import torch
import torch.nn as nn
import torch.optim as optim
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '7'
from tqdm import trange
from config import config
from utils import get_data_package, convert, get_alphabet
from model import CLIP
from dataset import lmdbDataset, resizeNormalize
import torchvision.transforms as transforms
from PIL import Image

# class resizeNormalize(object):

#     def __init__(self, size, interpolation=Image.BILINEAR, test=False):
#         self.size = size
#         self.test = test
#         self.interpolation = interpolation
#         self.toTensor = transforms.ToTensor()

#     def __call__(self, img):
#         img = img.resize(self.size, self.interpolation)
#         img = self.toTensor(img)
#         img.sub_(0.5).div_(0.5)
#         return img

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


transform = resizeNormalize((config['imageW'],config['imageH']))

alphabet = get_alphabet()

model = CLIP(embed_dim=2048, context_length=30, 
            vocab_size=len(alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()

model = nn.DataParallel(model)

def get_textfeatures_from_txt(model, txt_path):

    char_file = open(txt_path, 'r').read()
    char_list = list(char_file)

    # 初始化测试字符集
    tmp_text = convert(char_list)
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
    return char_list, text_features

def get_textfeatures_from_string(model, string):
    
    char_list = list(string)

    # 初始化测试字符集
    tmp_text = convert(char_list)
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
    return char_list, text_features

def infer_one_img(image, model, char_list, text_features, mode='largest', topk=10):

    image = transform(image)
    image = image.unsqueeze(dim=0)
    image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

    image = image.cuda()
    
    image_features, text_features, logit_scale = model(image, text_features, test=True)
    logits_per_image = logit_scale.item() * image_features @ text_features.t()

    image_features, text_features, logit_scale = model(image, text_features, test=True)
    # logits_per_image = logit_scale.tolist()[0] * image_features @ text_features.t()
    # probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)

    # print(logits_per_image.shape)
    if mode == 'largest':
        probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
        res = []
        # print(index)
        for i in range(len(image)):
            res.append(char_list[index[i]])

    elif mode == 'topk':
        logits_per_image = logits_per_image.softmax(dim=-1)
        probs, index = torch.topk(logits_per_image, topk, dim=-1, largest=True)
        res = []
        for i in range(len(image)):
            temp_res = []
            for prob, idx in zip(probs[i], index[i]):
                temp_res.append((char_list[idx], prob.item()))
            res.append(temp_res)

    return res


def test():

    alphabet_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/data/char_3755.txt'
    ckpt_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/history/clip_resume_Acient+doc+web+scene+scut+ctw+ic13/best_model.pth'
    test_ds_path = '/mnt/disk1/luwei/Archives_char_recognition/dataset/archives_test'

    # 模型加载
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 构建clip的搜索空间
    char_list, text_features = get_textfeatures_from_txt(model, alphabet_path)

    # 获得测试数据集
    test_ds = lmdbDataset(test_ds_path, resizeNormalize((config['imageW'], config['imageH']), test=False))
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config['batch'], shuffle=True, num_workers=8,)


    print("Start Eval!")
    model.eval()
    test_dataloader = iter(test_loader)
    test_loader_len = len(test_loader)

    with torch.no_grad():
        correct = 0
        total = 0
        with trange(test_loader_len) as t:
            for iteration in t:
                data = test_dataloader.next()
                image, label = data
                image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
                image = image.cuda()
                image_features, text_features, logit_scale = model(image, text_features, test=True)
                logits_per_image = logit_scale.item() * image_features @ text_features.t()
                probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
                for i in range(len(label)):
                    if traditional_to_simplify(char_list[index[i]]) == traditional_to_simplify(label[i]):
                        correct += 1
                    total += 1
                t.set_description('{}/{}'.format(iteration, test_loader_len))
                t.set_postfix(acc=correct/total)
        print(total)
        print("ACC : {}".format(correct / total))
        

def infer_Archives():
    import os, sys, json

    ckpt_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/history/clip_resume_Acient+doc+web+scene+scut+ctw+ic13/best_model.pth'
    exp_name = '6.18-paddle字符集-档案馆8张图所有字符'  # 3755测试 /  paddle字符集测试 / 全集测试
    alphabet_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/data/paddle_only_chinese.txt'

    # 模型加载
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 构建clip搜索空间
    char_list, text_features = get_textfeatures_from_txt(model, alphabet_path)

    img_root = '/mnt/disk1/luwei/Archives_char_recognition/img_crops'
    big_img_names = ['B172-1-72-7', 'D2-0-917_watermark', 'Snipaste_2024-05-02_11-53-42', 
                'Snipaste_2024-05-02_18-58-32', 'Snipaste_2024-05-02_18-59-28', '微信图片_20240502115242',
                '微信图片_20240502115258', '微信图片_20240502185952']

    save_root = '/home/luwei/code/archives_char_recognition/CCR_CLIP/predict_res/'
    save_root = os.path.join(save_root, exp_name)
    os.makedirs(save_root, exist_ok=True)

    for big_img_name in big_img_names:
        res = {}
        img_crop_root = os.path.join(img_root, big_img_name)
        for img_name in os.listdir(img_crop_root):
            img_path = os.path.join(img_crop_root, img_name)
            img = Image.open(img_path)
            
            result = infer_one_img(img, model, char_list, text_features)
            print(result)
            res[img_name] = result[0]

        save_path = os.path.join(save_root, f'{big_img_name}.json') 

        with open(save_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False)

        print(f'{big_img_name} is done!')


def infer_Archives_with_topk():
    import os, sys, json

    ckpt_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/history/clip_resume_Acient+doc+web+scene+scut+ctw+ic13/best_model.pth'
    exp_name = '6.18-paddle字符集+top10-档案馆8张图所有字符'  # 3755测试 /  paddle字符集测试 / 全集测试
    alphabet_path = '/home/luwei/code/archives_char_recognition/CCR_CLIP/data/paddle_only_chinese.txt'

    # 模型加载
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    # 构建clip搜索空间
    char_list, text_features = get_textfeatures_from_txt(model, alphabet_path)

    img_root = '/mnt/disk1/luwei/Archives_char_recognition/img_crops'
    big_img_names = ['B172-1-72-7', 'D2-0-917_watermark', 'Snipaste_2024-05-02_11-53-42', 
                'Snipaste_2024-05-02_18-58-32', 'Snipaste_2024-05-02_18-59-28', '微信图片_20240502115242',
                '微信图片_20240502115258', '微信图片_20240502185952']

    save_root = '/home/luwei/code/archives_char_recognition/CCR_CLIP/predict_res/'
    save_root = os.path.join(save_root, exp_name)
    os.makedirs(save_root, exist_ok=True)

    for big_img_name in big_img_names:
        res = {}
        img_crop_root = os.path.join(img_root, big_img_name)

        for img_name in os.listdir(img_crop_root):
            img_path = os.path.join(img_crop_root, img_name)
            img = Image.open(img_path).convert('RGB')
            
            result = infer_one_img(img, model, char_list, text_features, mode='topk', topk=10)
            print(result)
            res[img_name] = result[0]

        save_path = os.path.join(save_root, f'{big_img_name}.json') 

        with open(save_path, 'w') as f:
            json.dump(res, f, ensure_ascii=False)

        print(f'{big_img_name} is done!')

if __name__ == '__main__':

    infer_Archives()

    
    
    