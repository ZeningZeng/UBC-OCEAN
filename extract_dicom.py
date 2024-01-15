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
        self.transform = T.Compose([T.ToTensor(), T.Resize((224, 224), antialias=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
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


label_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'Other']
num_classes = len(label_names)
label_dict = {label_names[i]: i for i in range(len(label_names))}
train_csv = pd.read_csv('/wsi/beifen.csv')
model_type = 8
feature_dir = f'/home/data1/zzn/Projects/kaggle/wsi/wsi_feature_dir_vit_p{model_type}'
pretrained = f'/home/data1/zzn/Projects/kaggle/dino_vit_small_patch{model_type}_ep200.torch'
external_data_dir = '/home/data1/zzn/Projects/kaggle/wsi/external'
device = torch.device('cuda:1')

os.makedirs(feature_dir, exist_ok=True)

extractor = VisionTransformer(patch_size=model_type, embed_dim=384, num_heads=6, num_classes=0)
extractor.load_state_dict(torch.load(pretrained, map_location='cpu'))
extractor = extractor.to(device)
extractor.eval()

with torch.no_grad():
    for class_name in os.listdir(external_data_dir):
        if class_name not in label_names:
            continue

        class_idx = label_dict[class_name]

        for name in os.listdir(os.path.join(external_data_dir, class_name)):
            if name + '.pt' in os.listdir(feature_dir):
                continue
            start = time.time()

            dicomdataset = SingleDICOMDataset(os.path.join(external_data_dir, class_name), name, 512)
            loader = DataLoader(dicomdataset, 1024, False, num_workers=0, pin_memory=True)

            features = []

            for _, tiles in enumerate(loader):
                tiles = tiles.to(device, non_blocking=True)
                features.append(extractor(tiles).cpu())

                gc.collect()
                torch.cuda.empty_cache()

            features = torch.cat(features)
            torch.save(features, os.path.join(feature_dir, name + '.pt'))

            if int(name) not in list(train_csv['image_id']):
                train_csv.loc[len(train_csv)] = [int(name), class_name, dicomdataset.wsi.dimensions[0], dicomdataset.wsi.dimensions[1], False]

            dicomdataset.wsi.close()
            end = time.time()
            print('finish', name, class_name, len(dicomdataset), end - start)

            train_csv.to_csv('/home/data1/zzn/Projects/kaggle/wsi/extrain.csv', index=False)
