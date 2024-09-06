from config import config
dict_file = open(config['decompose_path'], 'r').readlines()
char_radical_dict = []
for line in dict_file:
    line = line.strip('\n')
    try:
        char, r_s = line.split(':')
    except:
        char, r_s = ':', ':'
    char_radical_dict.append(char)

save_path = '/home/luwei/code/FudanOCR-main/image-ids-CTR/CCR_CLIP/data/all_chinese.txt'
with open(save_path, 'w') as f:
    f.write(''.join(char_radical_dict))

