import torch
# import clip

import torch.nn as nn
import torch.optim as optim

from tqdm import trange
from config import config
from utils import get_data_package, convert, saver, get_alphabet, get_alphabet_s
# from model import convert_weights
from model_m import CLIP

# 建立模型
# device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
alphabet = get_alphabet()
alphabet_s = get_alphabet_s()
model = CLIP(embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
             vision_patch_size=32, context_length=config['max_len'],context_length_s=config['max_len_s'], vocab_size=len(alphabet), stroke_size=len(alphabet_s), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
model = nn.DataParallel(model)
if config['resume'] != '':
    model.load_state_dict(torch.load(config['resume']))
    print('加载成功')


# 读取数据
train_loader, test_loader = get_data_package()

loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-6)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# 读取所有的字符
char_file = open(config['char_path'], 'r').read()
char_3755 = list(char_file)

global best_acc, best_acc_s, best_acc_all
best_acc = -1
best_acc_s = -1
best_acc_all = -1

def val(model):
    print("Start Eval!")
    model.eval()
    test_dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('测试集长度:', test_loader_len)
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

    tmp_text, tmp_text_s, text_tensor_mask, stroke_tensor_mask, fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask = convert(char_3755)
    text_tensor_mask=text_tensor_mask.cuda()
    stroke_tensor_mask=stroke_tensor_mask.cuda()
    stroke_2_text_cross_attn_mask=stroke_2_text_cross_attn_mask.cuda()
    text_2_stroke_cross_attn_mask=text_2_stroke_cross_attn_mask.cuda()
    fused_tensor_mask=fused_tensor_mask.cuda()
    
    text_features = []
    text_features_s = []
    text_features_all = []
    iters = len(char_3755) // 100
    with torch.no_grad():
        for i in range(iters+1):
            s = i * 100
            e = (i + 1) * 100
            if e > len(char_3755):
                e = len(char_3755)
            if s == len(char_3755):
                break

            text_features_tmp, seq_text = model.module.encode_text(tmp_text[s:e],text_tensor_mask[s:e])
            text_features_tmp_s, seq_text_s = model.module.encode_text_s(tmp_text_s[s:e],stroke_tensor_mask[s:e])
            # print('seq_text:', seq_text.size())
            # print('seq_text_s:', seq_text_s.size())
            text_features_tmp_all = model.module.encode_all(seq_text, seq_text_s,text_2_stroke_cross_attn_mask[s:e],stroke_2_text_cross_attn_mask[s:e], fused_tensor_mask[s:e])
            text_features.append(text_features_tmp)
            text_features_s.append(text_features_tmp_s)
            text_features_all.append(text_features_tmp_all)
        text_features = torch.cat(text_features, dim=0)
        text_features_s = torch.cat(text_features_s, dim=0)
        text_features_all = torch.cat(text_features_all, dim=0)
        correct = 0
        correct_s = 0
        correct_all = 0
        total = 0
        with trange(test_loader_len) as t:
            for iteration in t:
                data = next(test_dataloader)
                image, label = data
                image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

                # image = image.to(device)
                image = image.cuda()
                image_features, text_features, text_features_s, text_features_all, logit_scale, logit_scale_s, logit_scale_all = \
                    model(image, text_features, text_features_s,text_tensor_mask, stroke_tensor_mask, fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask, text_features_fuse=text_features_all, test=True)
                logits_per_image = logit_scale[0] * image_features @ text_features.t()
                _, index = logits_per_image.softmax(dim=-1).max(dim=-1)

                logits_per_image_s = logit_scale_s[0] * image_features @ text_features_s.t()
                _, index_s = logits_per_image_s.softmax(dim=-1).max(dim=-1)

                logits_per_image_all = logit_scale_all[0] * image_features @ text_features_all.t()
                _, index_all = logits_per_image_all.softmax(dim=-1).max(dim=-1)

                for i in range(len(label)):
                    # print(char_3755[index[i]], '-->', label[i])
                    if char_3755[index[i]] == label[i]:
                        correct += 1
                    if char_3755[index_s[i]] == label[i]:
                        correct_s += 1
                    if char_3755[index_all[i]] == label[i]:
                        correct_all += 1
                    total += 1
                t.set_description('{}/{}'.format(iteration, test_loader_len))
                t.set_postfix(r_acc=correct/total, s_acc=correct_s/total, all_acc=correct_all/total)
        print("ACC_r : {}".format(correct / total))
        print("ACC_s : {}".format(correct_s / total))
        print("ACC_all : {}".format(correct_all / total))
        global best_acc, best_acc_s, best_acc_all
        # 保存最优模型
        if correct / total > best_acc:
            best_acc = correct / total
            torch.save(model.state_dict(), './history/{}/best_model_r.pth'.format(config['exp_name']))
        if correct_s / total > best_acc_s:
            best_acc_s = correct_s / total
            torch.save(model.state_dict(), './history/{}/best_model_s.pth'.format(config['exp_name']))
        if correct_all / total > best_acc_all:
            best_acc_all = correct_all / total
            torch.save(model.state_dict(), './history/{}/best_model_all.pth'.format(config['exp_name']))


    f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC_r : {} | ACC_s : {} | ACC_all : {}\n".format(epoch, correct / total, correct_s / total, correct_all / total))
    f.close()

saver()

for epoch in range(config['epoch']):
    dataloader = iter(train_loader)
    train_loader_len = len(train_loader)
    print('训练集长度:', train_loader_len)
    for iteration in range(train_loader_len):
        model.train()
        optimizer.zero_grad()

        data = next(dataloader)
        image, label = data

        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
        text, text_s, text_tensor_mask, stroke_tensor_mask, fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask = convert(label)

        # image = image.to(device)
        image = image.cuda()
        # text = text.to(device)
        text = text.cuda()
        text_s = text_s.cuda()
        # print('in main')
        # print(text_tensor_mask, stroke_tensor_mask, fused_tensor_mask)
        image_features, text_features, text_features_s, text_features_all, logit_scale, logit_scale_s, logit_scale_all = \
            model(image, text, text_s,text_tensor_mask, stroke_tensor_mask, fused_tensor_mask,text_2_stroke_cross_attn_mask,stroke_2_text_cross_attn_mask)

        logits_per_image = logit_scale[0] * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        logits_per_image_s = logit_scale_s[0] * image_features @ text_features_s.t()
        logits_per_text_s = logits_per_image_s.t()

        logits_per_image_all = logit_scale_all[0] * image_features @ text_features_all.t()
        logits_per_text_all = logits_per_image_all.t()
        label_str = ''.join(label)
        ground_truth = torch.arange(len(image), dtype=torch.long).cuda()

        for i in range(len(image)):
            ground_truth[i] = label_str.index(label[i])
        
        # print(text_features_all)
        # print(loss_img(logits_per_image_all, ground_truth))
        # print(loss_txt(logits_per_text_all, ground_truth))
        loss_r = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        loss_s = (loss_img(logits_per_image_s, ground_truth) + loss_txt(logits_per_text_s, ground_truth)) / 2
        loss_all = (loss_img(logits_per_image_all, ground_truth) + loss_txt(logits_per_text_all, ground_truth)) / 2

        total_loss = loss_r + 0.1 * loss_s + loss_all
        total_loss.backward()
        optimizer.step()

        print('epoch:{}, iter:{}/{}, loss_r:{}, loss_s:{}, loss_all:{}'.format(epoch, iteration, train_loader_len, loss_r, loss_s, loss_all))
        # val(model)
        # exit()
    val(model)
    if (epoch + 1) % 3 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.8
