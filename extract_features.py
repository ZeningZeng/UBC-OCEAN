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
        self.transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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
        tile = self.transform(tile)
        return tile


data_dir = '/home/data1/zzn/Projects/datasets/kaggle'
train_csv = pd.read_csv(os.path.join(data_dir, 'extrain.csv'))
model_type = 8
feature_dir = f'/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p{model_type}'
pretrained = f'/home/data1/zzn/Projects/kaggle/dino_vit_small_patch{model_type}_ep200.torch'
device = torch.device('cuda:2')

os.makedirs(feature_dir, exist_ok=True)

extractor = VisionTransformer(patch_size=model_type, embed_dim=384, num_heads=6, num_classes=0)
extractor.load_state_dict(torch.load(pretrained, map_location='cpu'))
extractor = extractor.to(device)
extractor.eval()

with torch.no_grad():
    for i in range(len(train_csv)):
        sample = train_csv.iloc[i]
        name = str(sample['image_id'])
        if name + '.pt' in os.listdir(feature_dir):
            continue
        start = time.time()

        wsidataset = SingleWSIDataset(data_dir, name, 512, 1.0, 'train')
        loader = DataLoader(wsidataset, 1024, False, num_workers=0, pin_memory=True)

        features = []

        for _, tiles in enumerate(loader):
            tiles = tiles.to(device, non_blocking=True)
            features.append(extractor(tiles).cpu())

            gc.collect()
            torch.cuda.empty_cache()

        features = torch.cat(features)
        torch.save(features, os.path.join(feature_dir, name + '.pt'))

        end = time.time()
        print('finish', name, end - start)
