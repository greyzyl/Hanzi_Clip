import pandas as pd
import torch
# import clip

import torch.nn as nn
import torch.optim as optim

from tqdm import trange
import tqdm
from config import config
from dataset import lmdbDataset, resizeNormalize
from utils import RandomIdentitySampler, convert2, get_data_package, convert, get_train_dataset_sampler, get_train_dataset_standard_char_sampler, saver, get_alphabet, get_alphabet_s
from model_m import CLIP
from torch.utils.data import Sampler, DataLoader
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
            data = next(gallery_dataloader)
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
def process_labels_confidences(labels, confidences):
    """
    处理多标签分类的标签和置信度：
    1. 对标签为1的位置对应的置信度进行平均。
    2. 将平均后的置信度放在标签第一次出现1的位置上。
    3. 合并标签，仅保留第一次出现的1，删除其余的1。
    
    参数：
    - labels (torch.Tensor): 一维张量，包含标签（0或1）。
    - confidences (torch.Tensor): 一维张量，包含对应的置信度。

    返回：
    - labels_new (torch.Tensor): 处理后的标签张量。
    - confidences_new (torch.Tensor): 处理后的置信度张量。
    """
    # 找出标签为1的位置索引
    indices = (labels == 1).nonzero(as_tuple=False).squeeze()

    if indices.numel() == 0:
        # 如果没有标签为1的情况，直接返回原始数据
        return labels, confidences

    # 计算这些位置对应的置信度的平均值
    avg_confidence = confidences[indices].mean()

    # 创建一个布尔掩码，表示要保留的位置（初始全部为True）
    mask = torch.ones_like(labels, dtype=torch.bool)

    # 将除第一次出现1的位置外的其他位置设为False（即删除）
    if indices.numel() > 1:
        mask[indices[1:]] = False

    # 应用掩码，得到新的标签和置信度张量
    labels_new = labels[mask]
    confidences_new = confidences[mask]

    # 更新第一次出现1的位置的置信度为平均值
    first_one_index = (labels_new == 1).nonzero(as_tuple=False)[0]
    confidences_new[first_one_index] = avg_confidence

    return labels_new, confidences_new

def merge_labels_and_confidences(labels, confidences):
    # labels和confidences是形状相同的torch张量
    n_rows, n_cols = labels.shape

    # 将labels转换为float类型以进行计算
    labels = labels.float()
    confidences = confidences.float()

    # 计算每一行标签为1的位置的总和（用于计算平均值）
    labels_sum = labels.sum(dim=1)  # Shape: (n_rows,)

    # 避免除以零的情况，如果某一行的标签总和为0，则将其设置为1以避免NaN
    labels_sum = labels_sum.masked_fill(labels_sum == 0, 1)

    # 计算每一行标签为1的位置的置信度总和
    confidences_weighted_sum = (confidences * labels).sum(dim=1)  # Shape: (n_rows,)

    # 计算平均置信度
    avg_confidences = confidences_weighted_sum / labels_sum  # Shape: (n_rows,)

    # 创建对角线的布尔掩码
    diag_mask = torch.eye(n_rows, n_cols, dtype=torch.bool, device=labels.device)

    # 创建标签为1的布尔掩码
    labels_bool = labels.bool()

    # 创建非对角线且标签为1的布尔掩码
    off_diag_mask = labels_bool & ~diag_mask

    # 将非对角线且标签为1的位置的置信度设置为-500
    confidences = confidences.masked_fill(off_diag_mask, -500.0)

    # 将非对角线且标签为1的位置的标签设置为0
    labels = labels.masked_fill(off_diag_mask, 0.0)

    # 将对角线位置的置信度设置为平均置信度
    confidences = confidences.masked_scatter(diag_mask, avg_confidences)

    # 确保对角线位置的标签为1
    labels = labels.masked_fill(diag_mask, 1.0)

    return labels, confidences
# 建立模型
# device = "cuda:0" if torch.cuda.is_available() else "cpu"  # If using GPU then use mixed precision training.
# model, preprocess = clip.load("ViT-B/32", device=device, jit=False)  # Must set jit=False for training
alphabet = get_alphabet()
alphabet_s = get_alphabet_s()

model = CLIP(config['vision_type'],embed_dim=2048, image_resolution=224, vision_layers=12, vision_width=768,
             vision_patch_size=32, context_length=config['max_len'], vocab_size=len(alphabet), stroke_size=len(alphabet_s), transformer_width=512,
             transformer_heads=8, transformer_layers=12).cuda()
model = nn.DataParallel(model)
if config['resume'] != '':
    model.load_state_dict(torch.load(config['resume']))
    print('加载成功')


# 读取数据

train_loader, test_loader = get_data_package()
# train_dataset=get_train_dataset_sampler()
loss_img = nn.CrossEntropyLoss()
loss_txt = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=config['lr'], betas=(0.9, 0.98), eps=1e-6)  # Params used from paper, the lr is smaller, more safe for fine tuning to new dataset

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

# 读取所有的字符
char_file = open(config['char_alphabet_path'], 'r').read()
char_3755 = list(char_file)

global best_acc, best_acc_s, best_acc_all
best_acc_r=-1
best_acc = -1
best_acc_s = -1
best_acc_all = -1
best_acc_i2i=-1
best_acc_combine=-1
def val(model):
    print("Start Eval!")

    model.eval()
    test_dataloader = iter(test_loader)
    test_loader_len = len(test_loader)
    print('test:', test_loader_len)
    torch.save(model.state_dict(), './history/{}/model.pth'.format(config['exp_name']))

    tmp_text, tmp_text_s= convert2(char_3755)
    text_features = []
    text_features_s=[]
    text_features_all= []
    iters = len(char_3755) // 100
    with torch.no_grad():

        #构造img gallery
        gallery_lmdb_path,imageW,imageH=config['standard_char_dataset'],config['imageW'],config['imageH']
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

        with trange(test_loader_len) as t:
            for iteration in t:
                data = next(test_dataloader)
                image, label = data
                image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))

                image = image.cuda()
                text_features_t=text_features.clone()
                image_features, text_features, text_features_s, text_features_all, logit_scale, logit_scale_s, logit_scale_all , logit_scale_i2i= model(image, text_features, text_features_s, text_features_all, test=True)

                #单独获取归一化后的标准字库图像特征
                standard_char_features=standard_char_features/standard_char_features.norm(dim=1, keepdim=True)
                
                '''
                计算real 2 ids的结果
                '''
                logits_per_image_r = logit_scale.tolist()[0] * image_features @ text_features.t()  # logit_scale.tolist() logit_scale.item() 
                probs_r, index_r = logits_per_image_r.softmax(dim=-1).max(dim=-1)

                logits_per_image_s = logit_scale_s.tolist()[0] * image_features @ text_features_s.t()
                probs_s, index_s = logits_per_image_s.softmax(dim=-1).max(dim=-1)

                logits_per_image_all = logit_scale_all.tolist()[0] * image_features @ text_features_all.t()
                probs_all, index_all = logits_per_image_all.softmax(dim=-1).max(dim=-1)

                '''
                计算real 2 standard char的结果
                '''
                logits_per_image_i2i = logit_scale_i2i.tolist()[0] * image_features @ standard_char_features.t()
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
                # t.set_postfix(acc_t2i=correct_t2i/total)
                # t.set_postfix(acc_combine=correct_combine/total)
                if total > 50:
                    break
        print(total)
        print("ACC : {}".format(max([correct_i2i,correct_r,correct_s,correct_all,correct_combine]) / total))
        global best_acc_combine
        global best_acc_i2i
        global best_acc_r
        global best_acc_s
        global best_acc_all
        if correct_i2i / total > best_acc_i2i:
            best_acc_i2i = correct_i2i / total
            torch.save(model.state_dict(), './history/{}/best_model_i2i.pth'.format(config['exp_name']))
        if correct_r / total > best_acc_r:
            best_acc_r = correct_r / total
            torch.save(model.state_dict(), './history/{}/best_model_r.pth'.format(config['exp_name']))
        if correct_s / total > best_acc_s:
            best_acc_s = correct_s / total
            torch.save(model.state_dict(), './history/{}/best_model_s.pth'.format(config['exp_name']))
        if correct_all / total > best_acc_all:
            best_acc_all = correct_all / total
            torch.save(model.state_dict(), './history/{}/best_model_all.pth'.format(config['exp_name']))        
        if correct_combine / total > best_acc_combine:
            best_acc_combine = correct_combine / total
            torch.save(model.state_dict(), './history/{}/best_model_combine.pth'.format(config['exp_name']))
    f = open('./history/{}/record.txt'.format(config['exp_name']), 'a+', encoding='utf-8')
    f.write("Epoch : {} | ACC : {},ACC_i2i:{},ACC_r:{},ACC_s:{},ACC_all:{},ACC_combine:{}\n".format(epoch, max([correct_i2i,correct_r,correct_s,correct_all,correct_combine]) / total,correct_i2i / total,correct_r / total,correct_s / total,correct_all / total,correct_combine / total))
    f.close()

if config['test_only']==True:
    val(model)
    exit()
saver()

for epoch in range(config['epoch']):
    # sampler = RandomIdentitySampler(train_dataset, batch_size=config['batch'],num_instances=1)
    # train_loader = DataLoader(train_dataset,batch_size=config['batch'], sampler=sampler, num_workers=8,
    #             pin_memory=True, drop_last=True,)
    dataloader = iter(train_loader)
    train_loader_len = len(train_loader)
    print('训练集长度:', train_loader_len)
    for iteration in range(train_loader_len):
        model.train()
        optimizer.zero_grad()

        data = next(dataloader)
        image, label = data

        image = torch.nn.functional.interpolate(image, size=(config['imageH'], config['imageW']))
        text, text_s = convert2(label)

        # image = image.to(device)
        image = image.cuda()
        # text = text.to(device)
        text = text.cuda()
        text_s = text_s.cuda()
        image_features, text_features, text_features_s, text_features_all, logit_scale, logit_scale_s, logit_scale_all, logit_scale_i2i = model(image, text, text_s)

        logits_per_image = logit_scale[0] * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()

        logits_per_image_s = logit_scale_s[0] * image_features @ text_features_s.t()
        logits_per_text_s = logits_per_image_s.t()

        logits_per_image_all = logit_scale_all[0] * image_features @ text_features_all.t()
        logits_per_text_all = logits_per_image_all.t()

        # 创建 ground_truth 矩阵
        # label_str = ''.join(label)
        label_encoded = pd.factorize(label)[0] 
        label_tensor = torch.tensor(label_encoded).cuda()
        label_tensor = label_tensor.unsqueeze(1)  # 转换为列向量
        ground_truth_t = (label_tensor == label_tensor.T).float()  # 比较生成 ground_truth 矩阵
        ground_truth, logits_per_image=merge_labels_and_confidences(ground_truth_t,logits_per_image)
        ground_truth, logits_per_text =merge_labels_and_confidences(ground_truth_t,logits_per_text)
        ground_truth, logits_per_image_s=merge_labels_and_confidences(ground_truth_t,logits_per_image_s)
        ground_truth, logits_per_text_s =merge_labels_and_confidences(ground_truth_t,logits_per_text_s)
        ground_truth, logits_per_image_all=merge_labels_and_confidences(ground_truth_t,logits_per_image_all)
        ground_truth, logits_per_text_all =merge_labels_and_confidences(ground_truth_t,logits_per_text_all)        

        loss_r = (loss_img(logits_per_image, ground_truth) + loss_txt(logits_per_text, ground_truth)) / 2
        loss_s = (loss_img(logits_per_image_s, ground_truth) + loss_txt(logits_per_text_s, ground_truth)) / 2
        loss_all = (loss_img(logits_per_image_all, ground_truth) + loss_txt(logits_per_text_all, ground_truth)) / 2

        total_loss = loss_r + 0.1 * loss_s + loss_all
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        print('epoch:{}, iter:{}/{}, loss_r:{}, loss_s:{}, loss_all:{}'.format(epoch, iteration, train_loader_len, loss_r, loss_s, loss_all))
        # val(model)
        # exit()
        if (iteration + 1) % (train_loader_len//5) == 0:
            val(model)
    # val(model)
    if (epoch + 1) > 10 and (epoch + 1) % 2 == 0:
        for p in optimizer.param_groups:
            p['lr'] *= 0.8
