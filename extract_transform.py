import os
import gc
import pandas as pd
import numpy as np
import pyvips
import random
import time

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import VisionTransformer

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 5_000_000_000
os.environ['VIPS_CONCURRENCY'] = '4'
os.environ['VIPS_DISC_THRESHOLD'] = '15gb'


class SingleWSIDataset(Dataset):
    def __init__(self, data_path: str, wsi_name: str, patch_size: int, ratio, mode: str):
        super().__init__()
        self.data_path = data_path
        self.wsi_name = wsi_name
        self.ratio = ratio
        assert mode in ['train', 'test']
        self.mode = mode
        self.wsi = pyvips.Image.new_from_file(os.path.join(data_path, f'{mode}_images', wsi_name + '.png'))
        self.img = Image.open(os.path.join(data_path, f'{mode}_images', wsi_name + '.png'))
        self.is_tma = self.wsi.height < 5000 and self.wsi.width < 5000
        self.patch_size = patch_size
        self.transform = T.Compose([T.RandomHorizontalFlip(),
                                    T.RandomVerticalFlip(),
                                    T.AutoAugment(),
                                    T.ToTensor(),
                                    T.Resize((224, 224), antialias=True),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.cor_list = self.get_patch()

    def get_patch(self):
        cor_list = []
        if self.is_tma:
            thumbnail = self.wsi
        else:
            thumbnail = pyvips.Image.new_from_file(os.path.join(self.data_path, f'{self.mode}_thumbnails', self.wsi_name + '_thumbnail.png'))
        wsi_width, wsi_height = self.wsi.width, self.wsi.height
        thu_width, thu_height = thumbnail.width, thumbnail.height
        h_r, w_r = wsi_height / thu_height, wsi_width / thu_width
        down_h, down_w = int(self.patch_size / h_r), int(self.patch_size / w_r)
        cors = [(x, y) for y in range(0, thu_height, down_h) for x in range(0, thu_width, down_w)]
        for x, y in cors:
            tile = thumbnail.crop(x, y, min(down_w, thu_width - x), min(down_h, thu_height - y)).numpy()[..., :3]
            black_bg = np.mean(tile, axis=2) < 20
            tile[black_bg, :] = 255
            mask_bg = np.mean(tile, axis=2) > 235
            if np.sum(mask_bg) < min(down_h, thu_height - y) * min(down_w, thu_width - x) * 0.5 or self.is_tma:
                cor_list.append((int(x * w_r), int(y * h_r)))

        random.shuffle(cor_list)
        if 0 < self.ratio <= 1:
            cor_list = cor_list[:max(int(len(cor_list) * self.ratio), 1)]
        elif self.ratio > 1:
            cor_list = cor_list[:min(len(cor_list), self.ratio)]
        return cor_list

    def __len__(self):
        return len(self.cor_list)

    def __getitem__(self, idx):
        x, y = self.cor_list[idx]
        tile = self.img.crop((x, y, min(x + self.patch_size, self.img.width - 1), min(y + self.patch_size, self.img.height - 1)))
        # import matplotlib.pyplot as plt
        # plt.imshow(tile)
        # plt.show()
        tile = self.transform(tile)
        return tile


label_names = ['CC', 'EC', 'LGSC', 'MC', 'Other']
num_classes = len(label_names)
label_dict = {label_names[i]: i for i in range(len(label_names))}
train_csv = pd.read_csv('/home/data1/zzn/Projects/kaggle/wsi/temp.csv')
feature_dir_p16 = '/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p16'
pretrained_p16 = '/home/data1/zzn/Projects/kaggle/dino_vit_small_patch16_ep200.torch'
feature_dir_p8 = '/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p8'
pretrained_p8 = '/home/data1/zzn/Projects/kaggle/dino_vit_small_patch8_ep200.torch'
external_data_dir = '/home/data1/zzn/Projects/datasets/kaggle'
device = torch.device('cuda:3')
times = {'CC': 4, 'EC': 3, 'HGSC': 0, 'LGSC': 10, 'MC': 10, 'Other': 10}

vit_p16 = VisionTransformer(patch_size=16, embed_dim=384, num_heads=6, num_classes=0)
vit_p16.load_state_dict(torch.load(pretrained_p16, map_location='cpu'))
# vit_p16 = torch.nn.DataParallel(vit_p16)
vit_p16 = vit_p16.to(device)
vit_p16.eval()
vit_p8 = VisionTransformer(patch_size=8, embed_dim=384, num_heads=6, num_classes=0)
vit_p8.load_state_dict(torch.load(pretrained_p8, map_location='cpu'))
# vit_p8 = torch.nn.DataParallel(vit_p8)
vit_p8 = vit_p8.to(device)
vit_p8.eval()

with torch.no_grad():
    for i in range(len(train_csv)):
        sample = train_csv.iloc[i]
        name = str(sample['image_id'])
        class_name = sample['label']
        if class_name not in label_names:
            continue

        wsidataset = SingleWSIDataset(external_data_dir, name, 512, 1.0, 'train')

        for t in range(times[class_name]):
            real_name = '8' + str(label_dict[class_name]) + name.zfill(5) + str(t)

            if real_name + '.pt' in os.listdir(feature_dir_p16) and real_name + '.pt' in os.listdir(feature_dir_p8):
                continue

            start = time.time()

            loader = DataLoader(wsidataset, 1024, False, num_workers=0, pin_memory=True)

            features_p16 = []
            features_p8 = []
            for _, tiles in enumerate(loader):
                tiles = tiles.to(device, non_blocking=True)
                features_p16.append(vit_p16(tiles).cpu())
                features_p8.append(vit_p8(tiles).cpu())
                gc.collect()
                torch.cuda.empty_cache()
            features_p16 = torch.cat(features_p16)
            features_p8 = torch.cat(features_p8)
            torch.save(features_p16, os.path.join(feature_dir_p16, real_name + '.pt'))
            torch.save(features_p8, os.path.join(feature_dir_p8, real_name + '.pt'))

            if int(real_name) not in list(train_csv['image_id']):
                train_csv.loc[len(train_csv)] = [int(real_name), class_name, wsidataset.wsi.width, wsidataset.wsi.height, False]

            end = time.time()
            print('finish', real_name, class_name, len(wsidataset), end - start)

            train_csv.to_csv('/home/data1/zzn/Projects/kaggle/wsi/temp.csv', index=False)
