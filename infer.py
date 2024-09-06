import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from config import config
from utils import get_data_package, convert, saver, get_alphabet
from model import CLIP
import torchvision.transforms as transforms
from PIL import Image

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


transform = resizeNormalize((config['imageW'],config['imageH']))

alphabet = get_alphabet()
# model = CLIP(embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
#              vision_patch_size=32, context_length=30, vocab_size=len(alphabet), transformer_width=512,
#              transformer_heads=8, transformer_layers=12).cuda()
model = CLIP(embed_dim=2048, context_length=30, 
            vocab_size=len(alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
model = nn.DataParallel(model)

model.load_state_dict(torch.load('/home/luwei/code/image-ids-CTR/CCR_CLIP/history/clip_resume_Acient+doc+web+scene+scut+ctw+ic13/best_model.pth'))


# char_file = open('./data/char_3755.txt', 'r').read()
# char_3755 = list(char_file)


# global best_acc
# best_acc = -1

# 初始化测试字符集


def get_textfeatures(texts):
    tmp_text = convert(texts)
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

def inference(image, text_features, texts):
    model.eval()
    image = transform(image)
    image = image.unsqueeze(dim=0)
    image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

    image = image.cuda()
    
    image_features, text_features, logit_scale = model(image, text_features, test=True)
    logits_per_image = logit_scale.item() * image_features @ text_features.t()
    probs, index = logits_per_image.softmax(dim=-1).max(dim=-1)
    res = []
    for i in range(len(image)):
        res.append(texts[index[i]])
    
    return res

def test2():
    img_root='/home/zhuyinglian/字符检测/craft-text-detector-master/output'
    json_root='/home/luwei/code/Stroke_Radical/rgb_img/radical_output_finetune_v2_mid'
    big_img_names = os.listdir(img_root)
    bad_case=[]
    for big_img_name in big_img_names:
        reses={}
        with open(os.path.join(json_root,big_img_name+'.json'),'r',encoding='utf-8') as f:
            js=json.load(f)
        for cut_img_name in js.keys():
            try:
                texts=js[cut_img_name]
                texts, text_features = get_textfeatures(texts)
                img_path=os.path.join(img_root,big_img_name,'crops_raw',cut_img_name)
                img = Image.open(img_path).convert('RGB')
                res = inference(img, text_features, texts)
                reses[cut_img_name]=res[0]
            except:
                bad_case.append((big_img_name,cut_img_name,texts))
                print(big_img_name,cut_img_name,texts)
        with open(f'workdir/{big_img_name}.json','w',encoding='utf-8') as f:
            json.dump(reses,f,ensure_ascii=False,indent=4)
    return bad_case
if __name__ == '__main__':
    bad=test2()
    print(bad)
    print(len(bad))
    # 获得text_features
    # texts = '工大土'
    # texts, text_features = get_textfeatures(texts)

    # # 读取图片
    # img_path = '/home/luwei/code/image-ids-CTR/微信图片_20240603182408.png'
    # img = Image.open(img_path).convert('RGB')

    # # 推理
    # res = inference(img, text_features, texts)

    # print(res)

    # img_root = '/home/zhuyinglian/字符检测/craft-text-detector-master/output/Snipaste_2024-05-02_11-53-42/crops_raw'

    # for img_name in os.listdir(img_root):
    #     img_path = os.path.join(img_root, img_name)
    #     img = Image.open(img_path)
    #     res = infer(img, text_features)
    #     print(res)
    