import os

import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from config import config
from utils import get_data_package, convert, saver, get_alphabet
from model import CLIP

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


alphabet = get_alphabet()
# model = CLIP(embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
#              vision_patch_size=32, context_length=30, vocab_size=len(alphabet), transformer_width=512,
#              transformer_heads=8, transformer_layers=12).cuda()
model = CLIP(embed_dim=2048, context_length=30, 
            vocab_size=len(alphabet), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
model = nn.DataParallel(model)


if config['resume'].strip() != '':
    model.load_state_dict(torch.load(config['resume']))
    print('loading！！！')


params_grad = [p.numel() for n, p in model.named_parameters()]
print(f"Number of Mapping Trainable Parameters: {sum(params_grad) / (1 << 20):.2f} M")

train_loader, test_loader = get_data_package()
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-6)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

char_file = open(config['char_path'], 'r').read()
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
            if e > len(char_file):
                e = len(char_file)
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
                    if traditional_to_simplify(char_3755[index[i]]) == traditional_to_simplify(label[i]):
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
