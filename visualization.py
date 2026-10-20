import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets.folder import *

from PIL import Image
import os
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import font_manager
import pickle

from model.pretrain_model_spaatt import Clip_Codebook
from utils.pretrain_utils_radicalrnn import *
from cfgs.train_config_rnn import config


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


class VisualizationDataset(ImageFolder):
    def __init__(self, root):
        super(VisualizationDataset, self).__init__(root, resizeNormalize((128, 128)))

    def __getitem__(self, index):
        # 调用 ImageFolder 的 __getitem__ 方法以获取图像数据和类别索引
        original_data = super(VisualizationDataset, self).__getitem__(index)
        # 获取文件夹名称作为标签
        img_path, _ = self.imgs[index]
        folder_name = os.path.basename(os.path.dirname(img_path))
        img = original_data[0]
        # img = img[0:1, :, :]
        label = int(folder_name[-5:])
        return img, label

    def find_classes(self, directory: str) -> Tuple[List[str], Dict[str, int]]:
        classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {directory}.")
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx

image_dir = './data/v6'

dataset = VisualizationDataset(image_dir)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

with open(config['char_dict'], 'rb') as f:
    char_dict = pickle.load(f)
    char_list = sorted(char_dict.keys())

alphabet = get_alphabet()
model = Clip_Codebook(vision_input_resolution=config['vision_input_resolution'],
                      vision_patch_size=config['vision_patch_size'],
                      vision_width=config['vision_width'],
                      vision_layers=config['vision_layers'],
                      vision_heads=config['vision_heads'],
                      vision_output_dim=config['vision_output_dim'],
                      text_embed_dim=config['text_embed_dim'],
                      text_context_length=config['max_len'],
                      text_vocab_size=len(alphabet),
                      text_width=config['text_width'],
                      text_heads=config['text_heads'],
                      text_layers=config['text_layers'],
                      codebook_num=config['codebook_num'],
                      codebook_dim=config['codebook_dim'],
                      temperature=config['temperature'],
                      is_threshold=config['is_threshold']).cuda()
# model.load_state_dict(torch.load('./hwdb1500rnn+cls+clip.pth', map_location='cuda:0'))
# model.load_state_dict(torch.load('./hwdb1500rnn.pth', map_location='cuda:0'))
# model.load_state_dict(torch.load('./hwdb1500rnn+spa.pth', map_location='cuda:0'))
model.load_state_dict(torch.load('./hwdb3755e16.pth', map_location='cuda:0'))


model.eval()

features = []
labels = []
with torch.no_grad():
    for image, label_idx in dataloader:
        image = image.cuda()

        global_features = model.visualization(image)

        features.append(global_features.cpu().numpy())
        labels.append(label_idx)

features = np.vstack(features)
labels = np.concatenate(labels)

# 使用t-SNE进行降维
tsne = TSNE(n_components=2, random_state=42, perplexity=150)
features_2d = tsne.fit_transform(features)

# 可视化
plt.figure(figsize=(16, 9))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels, cmap='tab10', s=70)

handles, _ = scatter.legend_elements()
unique_labels = np.unique(labels)
legend_labels = [char_list[i] for i in unique_labels]
zh_font = font_manager.FontProperties(fname='./msyh.ttc')

# 添加图例
legend1 = plt.legend(handles, legend_labels, prop=zh_font, fontsize=30, loc='center left', bbox_to_anchor=(1, 0.5))
plt.gca().add_artist(legend1)  # 使用当前的Axes对象调用add_artist
# plt.legend(handles, legend_labels, prop=zh_font, fontsize=12,
#            loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0, ncol=1, frameon=True, shadow=True)
# plt.legend(handles, legend_labels, prop=zh_font, fontsize='large')

plt.show()