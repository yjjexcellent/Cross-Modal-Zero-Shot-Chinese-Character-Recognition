config = {
    'epoch': 50,
    'train_dataset': './data/ttf_image_data/img/train/',
    # 'val_dataset': './data/ttf_image_data/img/val/',
    'val_dataset': './data/unseen/img/test/',
    'batch_size': 64,
    'alphabet_path': './data/IDS/radical_alphabet_3755.txt',
    # 'decompose_path': './data/IDS/decompose_3755.txt',
    'decompose_path': './data/IDS/unseen_decompose_1000.txt',
    'lr': 2e-5,
    # 'char_num': 3755,
    'char_num': 1000,
    'max_len': 30,
    'codebook_num': 200,
    'codebook_dim': 256
}