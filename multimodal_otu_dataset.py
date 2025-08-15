
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import random
from PIL import Image, ImageEnhance

class MultiModalOTUDataset(Dataset):
    def __init__(self, root_dir, split='train', augment=False):
        self.root_dir = root_dir
        self.augment = augment
        self.split = split.lower()

        # 加载 id 列表
        id_path = os.path.join(root_dir, f"{self.split}_ids.txt")
        with open(id_path, "r") as f:
            self.ids = [line.strip() for line in f.readlines()]

        # 构建图像路径
        self.x2d_dir = os.path.join(root_dir, "2d", "images")
        self.y2d_dir = os.path.join(root_dir, "2d", "masks")
        self.xceus_dir = os.path.join(root_dir, "3d", "images")
        self.yceus_dir = os.path.join(root_dir, "3d", "masks")  # 可选

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]

        x2d = np.load(os.path.join(self.x2d_dir, f"{img_id}.npy")).astype(np.float32)
        xceus = np.load(os.path.join(self.xceus_dir, f"{img_id}.npy")).astype(np.float32)
        mask = np.load(os.path.join(self.y2d_dir, f"{img_id}.npy")).astype(np.uint8)  # 使用 2D mask

        # 数据增强（2D + CEUS + mask 同时处理）
        if self.augment:
            x2d, xceus, mask = self._augment(x2d, xceus, mask)

        # 转换为 torch.Tensor，添加通道维度 [1,H,W]
        return (
            torch.tensor(x2d).unsqueeze(0),       # x2d: [1, H, W]
            torch.tensor(xceus).unsqueeze(0),     # xceus: [1, H, W]
            torch.tensor(mask).unsqueeze(0).float()  # mask: [1, H, W]
        )

    def _augment(self, x2d, xceus, y):
        img2d = Image.fromarray((x2d * 255).astype(np.uint8))
        imgc = Image.fromarray((xceus * 255).astype(np.uint8))
        msk = Image.fromarray((y * 255).astype(np.uint8))

        if random.random() < 0.5:
            img2d = img2d.transpose(Image.FLIP_LEFT_RIGHT)
            imgc = imgc.transpose(Image.FLIP_LEFT_RIGHT)
            msk = msk.transpose(Image.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img2d = img2d.rotate(angle)
            imgc = imgc.rotate(angle)
            msk = msk.rotate(angle)

        if random.random() < 0.5:
            imgc = ImageEnhance.Contrast(imgc).enhance(random.uniform(0.8, 1.2))

        x2d_aug = np.array(img2d).astype(np.float32) / 255.0
        xceus_aug = np.array(imgc).astype(np.float32) / 255.0
        y_aug = (np.array(msk) > 127).astype(np.uint8)

        return x2d_aug, xceus_aug, y_aug
