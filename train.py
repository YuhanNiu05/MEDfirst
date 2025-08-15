
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import f1_score
from multimodal_otu_dataset import MultiModalOTUDataset
from multimodal_segnet import MultiModalSegNet  # 你之前的模型

# === 参数设置 ===
root = "/root/autodl-tmp/preprocessed_mmOTU"
batch_size =8
lr = 1e-4
epochs = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# === 加载数据 ===
train_set = MultiModalOTUDataset(root, split="train", augment=True)
val_set = MultiModalOTUDataset(root, split="val", augment=False)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size)

# === 初始化模型 ===
model = MultiModalSegNet()
model.to(device)

# === 损失函数与优化器 ===
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

def dice_score(preds, targets, threshold=0.5):
    preds = (torch.sigmoid(preds) > threshold).float()
    smooth = 1e-6
    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean().item()

# === 训练过程 ===
best_dice = 0
os.makedirs("checkpoints1", exist_ok=True)

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0
    for x2d, xceus, y in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}"):
        x2d, xceus, y = x2d.to(device), xceus.to(device), y.to(device)
        output = model(x2d, xceus)
        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * x2d.size(0)

    # === 验证 ===
    model.eval()
    val_loss = 0
    val_dice = 0
    with torch.no_grad():
        for x2d, xceus, y in val_loader:
            x2d, xceus, y = x2d.to(device), xceus.to(device), y.to(device)
            output = model(x2d, xceus)
            loss = criterion(output, y)
            val_loss += loss.item() * x2d.size(0)
            val_dice += dice_score(output, y) * x2d.size(0)

    avg_train_loss = train_loss / len(train_set)
    avg_val_loss = val_loss / len(val_set)
    avg_val_dice = val_dice / len(val_set)
    print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f} | Val Loss={avg_val_loss:.4f} | Val Dice={avg_val_dice:.4f}")

    # === 保存最优模型 ===
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), "checkpoints1/best_model.pth")
        print("✅ Saved best model.")

print(f"🎯 Training complete. Best Dice = {best_dice:.4f}")
