config = {
    'epoch': 200,
    'train_dataset': ['/home/fdu02/fdu02_dir/zyl/data/hanziclip_data_lmdb/byte_guji_data/train/patch_MTHTKN','/home/fdu02/fdu02_dir/zyl/data/hanziclip_data_lmdb/byte_guji_data/train/patch_casia','/home/fdu02/fdu02_dir/zyl/data/hanziclip_data_lmdb/byte_guji_data/train/patch_SIKU'],
    # 'train_dataset': '/home/fdu02/fdu02_dir/zyl/data/CharacterZeroShot/train_500_1',
    'test_dataset': ['/home/fdu02/fdu02_dir/zyl/data/hanziclip_data_lmdb/byte_guji_data/test/v1_test','/home/fdu02/fdu02_dir/zyl/data/hanziclip_data_lmdb/byte_guji_data/test/SIKU_val_v1'],
    'batch': 64,
    'imageW': 128,
    'imageH': 128,
    'resume': '',
    'char_path':'./data/all_chinese.txt',
    'alphabet_path': './data/radical_27533.txt', # rsst_alphabet.txt radical_alphabet.txt
    'decompose_path': './data/decompose.txt', # rsst_decompose.txt decompose.txt
    'alphabet_path_s': './data/rsst_alphabet_r.txt', # rsst_alphabet.txt radical_alphabet.txt
    'decompose_path_s': './data/27533_rsst_decompose.txt', # rsst_decompose.txt decompose.txt
    'max_len': 30,
    'max_len_s':60,
    'lr': 1e-4,
    'exp_name': '【09-06】特征融合_部首30_笔画60_add_mask',
}