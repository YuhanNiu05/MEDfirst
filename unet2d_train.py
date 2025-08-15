
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm import tqdm

# ==== 模型结构 ====
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet2D(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.down1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.upconv2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.upconv1 = DoubleConv(128, 64)
        self.out_conv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(self.pool1(d1))
        bn = self.bottleneck(self.pool2(d2))
        u2 = self.up2(bn)
        u2 = self.upconv2(torch.cat([u2, d2], dim=1))
        u1 = self.up1(u2)
        u1 = self.upconv1(torch.cat([u1, d1], dim=1))
        return torch.sigmoid(self.out_conv(u1))

# ==== 数据集类 ====
class Ultrasound2DDataset(Dataset):
    def __init__(self, root_dir, id_txt):
        self.image_dir = os.path.join(root_dir, "images")
        self.mask_dir = os.path.join(root_dir, "masks")
        with open(id_txt, 'r') as f:
            self.ids = [line.strip() for line in f]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        image = np.load(os.path.join(self.image_dir, f"{id}.npy"))[None, :, :]
        mask = np.load(os.path.join(self.mask_dir, f"{id}.npy"))[None, :, :]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32)

# ==== 配置参数 ====
root = "/root/autodl-tmp/OTU_2d/processed_2d"
id_txt = "/root/autodl-tmp/OTU_2d/processed_2d/processed_ids.txt"
batch_size = 8
lr = 1e-3
num_epochs = 20
device = "cuda" if torch.cuda.is_available() else "cpu"

print("设备:", device)
print("超参数：batch_size =", batch_size, ", lr =", lr, ", epoch =", num_epochs)

# ==== 数据加载 ====
full_dataset = Ultrasound2DDataset(root, id_txt)
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_set, val_set = random_split(full_dataset, [train_size, val_size])
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# ==== 初始化模型 ====
model = UNet2D().to(device)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ==== 训练 ====
start_time = time.time()
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)
        loss = criterion(preds, masks)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train = train_loss / len(train_loader)

    model.eval()
    with torch.no_grad():
        val_loss = 0
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            val_loss += criterion(preds, masks).item()
        avg_val = val_loss / len(val_loader)

    print(f"Epoch {epoch+1}: Train Loss = {avg_train:.4f}, Val Loss = {avg_val:.4f}")

# ==== 训练时间统计 ====
elapsed = time.time() - start_time
print(f"训练总耗时：{elapsed:.2f} 秒")

# ==== 模型保存 ====
torch.save(model.state_dict(), "unet2d_model.pth")
print("✅ 模型保存至 unet2d_model.pth")
