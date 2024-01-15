import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

# 0 Raw data loading
data_dir = '/home/data1/zzn/Projects/datasets/kaggle'
train_csv = pd.read_csv(os.path.join('/home/data1/zzn/Projects/kaggle/wsi', 'extrain.csv'))
train_data = train_csv.iloc[np.r_[0:100, 100:536]].reset_index(drop=True)
val_data = train_csv.iloc[536:1709].reset_index(drop=True)
label_names = ['CC', 'EC', 'HGSC', 'LGSC', 'MC', 'Other']
num_classes = len(label_names)
label_dict = {label_names[i]: i for i in range(len(label_names))}
epochs = 15
in_dim = 384
model_type = 8
ratio = 1.0
mil_type = 'transmil'
accumulate = True
test = False
feature_dir = f'wsi_feature_dir_vit_p{model_type}'
mil_model_name = f'transform_wsi_vitp{model_type}_{mil_type}_{ratio}_{epochs}ep.pth'
device = torch.device('cuda:2')


# 1 Dataset and transform
class WSIFeatDataset(Dataset):
    def __init__(self, data_csv: pd.DataFrame, feature_dir: str, ratio, phase: int):
        super().__init__()
        self.data_csv = data_csv
        self.feature_dir = feature_dir
        self.ratio = ratio
        assert phase in [0, 1]
        self.phase = phase

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        sample = self.data_csv.iloc[idx]
        file_name = str(sample['image_id'])

        features = torch.load(os.path.join(self.feature_dir, file_name + '.pt'), map_location='cpu')
        random.shuffle(features)
        if 0 < self.ratio <= 1:
            features = features[:int(len(features) * self.ratio)]
        elif self.ratio > 1:
            features = features[:min(len(features), self.ratio)]

        if self.phase == 0:
            label = torch.tensor(label_dict[sample['label']])
            return file_name, features, label
        else:
            return file_name, features


# 2 Model
class ABMIL(nn.Module):
    def __init__(self, in_dim, feat_dim, attn_dim, num_classes):
        super().__init__()
        self.downlinear = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.ReLU())
        self.attention_V = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(feat_dim, attn_dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(attn_dim, 1)
        self.classifier = nn.Linear(feat_dim, num_classes)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.downlinear(x)

        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U)
        A = torch.transpose(A, 1, 0)
        A = torch.softmax(A, dim=1)
        x = torch.mm(A, x)

        scores = self.classifier(x)

        return scores


class FCLayer(nn.Module):
    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        x = self.fc(feats)
        return feats, x


class IClassifier(nn.Module):
    def __init__(self, feature_size, output_class):
        super(IClassifier, self).__init__()

        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, feats):
        c = self.fc(feats.view(feats.shape[0], -1))  # N x C
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self, input_size, output_class, dropout_v=0.0, nonlinear=True, passing_v=False):  # K, L, N
        super(BClassifier, self).__init__()
        if nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, 128), nn.ReLU(), nn.Linear(128, 128), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, 128)
        if passing_v:
            self.v = nn.Sequential(
                nn.Dropout(dropout_v),
                nn.Linear(input_size, input_size),
                nn.ReLU()
            )
        else:
            self.v = nn.Identity()

        ### 1D convolutional layer that can handle multiple class (including binary)
        self.fcc = nn.Conv1d(output_class, output_class, kernel_size=input_size)

    def forward(self, feats, c):  # N x K, N x C
        device = feats.device
        V = self.v(feats)  # N x V, unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # N x Q, unsorted

        # handle multiple classes without for loop
        _, m_indices = torch.sort(c, 0, descending=True)  # sort class scores along the instance dimension, m_indices in shape N x C
        m_feats = torch.index_select(feats, dim=0, index=m_indices[0, :])  # select critical instances, m_feats in shape C x K
        q_max = self.q(m_feats)  # compute queries of critical instances, q_max in shape C x Q
        A = torch.mm(Q, q_max.transpose(0, 1))  # compute inner product of Q to each entry of q_max, A in shape N x C, each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)), 0)  # normalize attention scores, A in shape N x C,
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation, B in shape C x V

        B = B.view(1, B.shape[0], B.shape[1])  # 1 x C x V
        C = self.fcc(B)  # 1 x C x 1
        C = C.view(1, -1)
        return C, A, B


class DSMIL(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        super(DSMIL, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        feats, classes = self.i_classifier(x)
        prediction_bag, A, B = self.b_classifier(feats, classes)

        return classes, prediction_bag, A, B


from einops import rearrange, reduce
from torch import einsum
from math import ceil


def exists(val):
    return val is not None


def moore_penrose_iter_pinv(x, iters=6):
    device = x.device

    abs_x = torch.abs(x)
    col = abs_x.sum(dim=-1)
    row = abs_x.sum(dim=-2)
    z = rearrange(x, '... i j -> ... j i') / (torch.max(col) * torch.max(row))

    I = torch.eye(x.shape[-1], device=device)
    I = rearrange(I, 'i j -> () i j')

    for _ in range(iters):
        xz = x @ z
        z = 0.25 * z @ (13 * I - (xz @ (15 * I - (xz @ (7 * I - xz)))))

    return z


class NystromAttention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=64,
            heads=8,
            num_landmarks=256,
            pinv_iterations=6,
            residual=True,
            residual_conv_kernel=33,
            eps=1e-8,
            dropout=0.
    ):
        super().__init__()
        self.eps = eps
        inner_dim = heads * dim_head

        self.num_landmarks = num_landmarks
        self.pinv_iterations = pinv_iterations

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

        self.residual = residual
        if residual:
            kernel_size = residual_conv_kernel
            padding = residual_conv_kernel // 2
            self.res_conv = nn.Conv2d(heads, heads, (kernel_size, 1), padding=(padding, 0), groups=heads, bias=False)

    def forward(self, x, mask=None, return_attn=False):
        b, n, _, h, m, iters, eps = *x.shape, self.heads, self.num_landmarks, self.pinv_iterations, self.eps

        # pad so that sequence can be evenly divided into m landmarks

        remainder = n % m
        if remainder > 0:
            padding = m - (n % m)
            x = F.pad(x, (0, 0, padding, 0), value=0)

            if exists(mask):
                mask = F.pad(mask, (padding, 0), value=False)

        # derive query, keys, values

        q, k, v = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # set masked positions to 0 in queries, keys, values

        if exists(mask):
            mask = rearrange(mask, 'b n -> b () n')
            q, k, v = map(lambda t: t * mask[..., None], (q, k, v))

        q = q * self.scale

        # generate landmarks by sum reduction, and then calculate mean using the mask

        l = ceil(n / m)
        landmark_einops_eq = '... (n l) d -> ... n d'
        q_landmarks = reduce(q, landmark_einops_eq, 'sum', l=l)
        k_landmarks = reduce(k, landmark_einops_eq, 'sum', l=l)

        # calculate landmark mask, and also get sum of non-masked elements in preparation for masked mean

        divisor = l
        if exists(mask):
            mask_landmarks_sum = reduce(mask, '... (n l) -> ... n', 'sum', l=l)
            divisor = mask_landmarks_sum[..., None] + eps
            mask_landmarks = mask_landmarks_sum > 0

        # masked mean (if mask exists)

        q_landmarks /= divisor
        k_landmarks /= divisor

        # similarities

        einops_eq = '... i d, ... j d -> ... i j'
        sim1 = einsum(einops_eq, q, k_landmarks)
        sim2 = einsum(einops_eq, q_landmarks, k_landmarks)
        sim3 = einsum(einops_eq, q_landmarks, k)

        # masking

        if exists(mask):
            mask_value = -torch.finfo(q.dtype).max
            sim1.masked_fill_(~(mask[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim2.masked_fill_(~(mask_landmarks[..., None] * mask_landmarks[..., None, :]), mask_value)
            sim3.masked_fill_(~(mask_landmarks[..., None] * mask[..., None, :]), mask_value)

        # eq (15) in the paper and aggregate values

        attn1, attn2, attn3 = map(lambda t: t.softmax(dim=-1), (sim1, sim2, sim3))
        attn2_inv = moore_penrose_iter_pinv(attn2, iters)

        out = (attn1 @ attn2_inv) @ (attn3 @ v)

        # add depth-wise conv residual of values

        if self.residual:
            out += self.res_conv(v)

        # merge and combine heads

        out = rearrange(out, 'b h n d -> b n (h d)', h=h)
        out = self.to_out(out)
        out = out[:, -n:]

        if return_attn:
            attn = attn1 @ attn2_inv @ attn3
            return out, attn

        return out


class TransLayer(nn.Module):
    def __init__(self, norm_layer=nn.LayerNorm, dim=512):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = NystromAttention(
            dim=dim,
            dim_head=dim // 8,
            heads=8,
            num_landmarks=dim // 2,  # number of landmarks
            pinv_iterations=6,  # number of moore-penrose iterations for approximating pinverse. 6 was recommended by the paper
            residual=True,  # whether to do an extra residual with the value or not. supposedly faster convergence if turned on
            dropout=0.1
        )

    def forward(self, x):
        x = x + self.attn(self.norm(x))

        return x


class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3 // 2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat) + cnn_feat + self.proj1(cnn_feat) + self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1), x), dim=1)
        return x


class TransMIL(nn.Module):
    def __init__(self, in_dim, n_classes):
        super(TransMIL, self).__init__()
        self.pos_layer = PPEG(dim=512)
        self._fc1 = nn.Sequential(nn.Linear(in_dim, 512), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, 512))
        self.n_classes = n_classes
        self.layer1 = TransLayer(dim=512)
        self.layer2 = TransLayer(dim=512)
        self.norm = nn.LayerNorm(512)
        self._fc2 = nn.Linear(512, self.n_classes)

    def forward(self, h):
        h = h.float()  # [B, n, 1024]

        h = self._fc1(h)  # [B, n, 512]

        # ---->pad
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:, :add_length, :]], dim=1)  # [B, N, 512]

        # ---->cls_token
        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).to(device)
        h = torch.cat((cls_tokens, h), dim=1)

        # ---->Translayer x1
        h = self.layer1(h)  # [B, N, 512]

        # ---->PPEG
        h = self.pos_layer(h, _H, _W)  # [B, N, 512]

        # ---->Translayer x2
        h = self.layer2(h)  # [B, N, 512]

        # ---->cls_token
        h = self.norm(h)[:, 0]

        # ---->predict
        logits = self._fc2(h)  # [B, n_classes]

        return logits


# 3 Training
train_dataset = WSIFeatDataset(train_data, feature_dir, ratio, 0) if test else WSIFeatDataset(train_csv, feature_dir, ratio, 0)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
val_dataset = WSIFeatDataset(val_data, feature_dir, ratio, 0)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)

if mil_type == 'abmil':
    model = ABMIL(in_dim, 512, 128, num_classes)
elif mil_type == 'dsmil':
    model = DSMIL(IClassifier(in_dim, num_classes), BClassifier(in_dim, num_classes))
elif mil_type == 'transmil':
    model = TransMIL(in_dim, num_classes)

optimizer = optim.Adam(model.parameters(), 5e-4, weight_decay=5e-4)
scheduler = lr_scheduler.CosineAnnealingLR(optimizer, epochs, 5e-5)

model = model.to(device)

max_acc = 0.
balanced_acc = 0.
for epoch in range(1, epochs + 1):
    loss_sum = 0.
    n = 0

    loop = tqdm(train_loader, total=len(train_loader))
    model.train()
    for file_name, features, label in loop:
        label = label.to(device)
        features = features.squeeze(0).to(device)

        if mil_type == 'abmil':
            scores = model(features)
            loss = F.cross_entropy(scores, label)
        elif mil_type == 'dsmil':
            classes, bag_prediction, _, _ = model(features)
            max_prediction, index = torch.max(classes, 0, True)
            loss_bag = F.cross_entropy(bag_prediction, label)
            loss_max = F.cross_entropy(max_prediction.view(1, -1), label)
            loss = 0.5 * loss_bag + 0.5 * loss_max
        elif mil_type == 'transmil':
            scores = model(features.unsqueeze(0))
            loss = F.cross_entropy(scores, label)

        if accumulate:
            loss = loss / 4
            loss.backward(retain_graph=True)
            if (n + 1) % 4 == 0 or (n + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
        else:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        n += 1
        loss_sum += loss.item()

        loop.set_description(f'Train [{epoch}/{epochs}]')
        loop.set_postfix(loss=loss.item(), loss_mean=loss_sum / n)

    if test:
        with torch.no_grad():
            acc = 0
            loss_val = 0
            y_true = []
            y_pred = []

            loop = tqdm(val_loader, total=len(val_loader))
            model.eval()
            for file_name, features, label in loop:
                label = label.to(device)
                features = features.squeeze(0).to(device)

                if mil_type == 'abmil':
                    scores = model(features)
                    scores = torch.softmax(scores, 1)
                elif mil_type == 'dsmil':
                    classes, bag_prediction, _, _ = model(features)
                    max_prediction, index = torch.max(classes, 0, True)
                    scores = 0.5 * torch.softmax(max_prediction, 1) + 0.5 * torch.softmax(bag_prediction, 1)
                elif mil_type == 'transmil':
                    scores = model(features.unsqueeze(0))
                    scores = torch.softmax(scores, 1)

                pred = torch.argmax(scores)

                y_pred.append(pred.item())
                y_true.append(label.item())

                if pred == label.squeeze(0):
                    acc += 1

                loop.set_description(f'Val [{epoch}/{epochs}]')
                loop.set_postfix(acc=acc / len(val_loader), max_acc=max_acc, balanced_acc=balanced_acc)

            if acc / len(val_loader) > max_acc:
                max_acc = acc / len(val_loader)
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            if balanced_accuracy_score(y_true, y_pred) > balanced_acc:
                balanced_acc = balanced_accuracy_score(y_true, y_pred)

    scheduler.step()

    if not test:
        torch.save(model.state_dict(), mil_model_name)
