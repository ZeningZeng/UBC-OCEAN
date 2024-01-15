import os
import gc
import pandas as pd
import numpy as np
import openslide
import time

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from timm.models.vision_transformer import VisionTransformer

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


class SingleDICOMDataset(Dataset):
    def __init__(self, data_path: str, wsi_name: str, patch_size: int):
        super().__init__()
        self.data_path = data_path
        self.wsi_name = wsi_name
        self.patch_size = patch_size
        self.wsi = openslide.OpenSlide(os.path.join(data_path, wsi_name))
        self.transform = T.Compose([T.RandomHorizontalFlip(),
                                    T.RandomVerticalFlip(),
                                    T.AutoAugment(),
                                    T.ToTensor(),
                                    T.Resize((224, 224), antialias=True),
                                    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        self.cor_list = self.get_patch()

    def get_patch(self):
        cor_list = []
        wsi_width, wsi_height = self.wsi.dimensions
        thumbnail = self.wsi.get_thumbnail((3000, int(3000 * wsi_height / wsi_width)))
        thu_width, thu_height = thumbnail.width, thumbnail.height
        h_r, w_r = wsi_height / thu_height, wsi_width / thu_width
        down_h, down_w = int(self.patch_size / h_r), int(self.patch_size / w_r)
        cors = [(x, y) for y in range(0, thu_height, down_h) for x in range(0, thu_width, down_w)]
        for x, y in cors:
            tile = np.array(thumbnail.crop((x, y, min(x + down_w, thu_width), min(y + down_h, thu_height))).convert('RGB'))
            black_bg = np.mean(tile, axis=2) < 20
            tile[black_bg, :] = 255
            mask_bg = np.mean(tile, axis=2) > 235
            if np.sum(mask_bg) < min(down_h, thu_height - y) * min(down_w, thu_width - x) * 0.5:
                cor_list.append((int(x * w_r), int(y * h_r)))

        return cor_list

    def __len__(self):
        return len(self.cor_list)

    def __getitem__(self, idx):
        x, y = self.cor_list[idx]
        tile = np.array(self.wsi.read_region((x, y), 0, (min(self.patch_size, self.wsi.dimensions[0] - x), min(self.patch_size, self.wsi.dimensions[1] - y))).convert('RGB'))
        # import matplotlib.pyplot as plt
        # plt.imshow(tile)
        # plt.show()
        tile = self.transform(tile)
        return tile


label_names = ['CC', 'EC', 'LGSC', 'MC', 'Other']
num_classes = len(label_names)
label_dict = {label_names[i]: i for i in range(len(label_names))}
train_csv = pd.read_csv('/home/data1/zzn/Projects/kaggle/wsi/extrain.csv')
feature_dir_p16 = '/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p16'
pretrained_p16 = '/home/data1/zzn/Projects/kaggle/dino_vit_small_patch16_ep200.torch'
feature_dir_p8 = '/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p8'
pretrained_p8 = '/home/data1/zzn/Projects/kaggle/dino_vit_small_patch8_ep200.torch'
external_data_dir = '/home/data1/zzn/Projects/kaggle/wsi/external'
device = torch.device('cuda:2')
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
    for class_name in os.listdir(external_data_dir):
        if class_name not in label_names:
            continue

        for name in os.listdir(os.path.join(external_data_dir, class_name)):
            dicomdataset = SingleDICOMDataset(os.path.join(external_data_dir, class_name), name, 512)

            for t in range(times[class_name]):
                real_name = '8' + name[1:] + str(t)

                if real_name + '.pt' in os.listdir(feature_dir_p16) and real_name + '.pt' in os.listdir(feature_dir_p8):
                    continue

                start = time.time()

                loader = DataLoader(dicomdataset, 1024, False, num_workers=0, pin_memory=True)

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
                    train_csv.loc[len(train_csv)] = [int(real_name), class_name, dicomdataset.wsi.dimensions[0], dicomdataset.wsi.dimensions[1], False]

                end = time.time()
                print('finish', real_name, class_name, len(dicomdataset), end - start)

                train_csv.to_csv('/home/data1/zzn/Projects/kaggle/wsi/1.csv', index=False)
