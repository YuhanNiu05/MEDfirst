
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image

from multimodal_segnet import MultiModalSegNet

# ==== 参数设置 ====
data_dir = "/root/autodl-tmp/preprocessed_mmOTU"
checkpoint_path = "checkpoints1/best_model.pth"
output_dir = "test_predictions"
batch_size = 4

os.makedirs(output_dir, exist_ok=True)

# ==== 数据集类 ====
class MultiModalOTUDataset(Dataset):
    def __init__(self, root, split="test", augment=False):
        self.root = root
        self.split = split
        self.augment = augment
        self.img_dir_2d = os.path.join(root, "2d", "images")
        self.img_dir_ceus = os.path.join(root, "3d", "images")
        self.mask_dir = os.path.join(root, "2d", "masks")
        with open(os.path.join(root, f"{split}_ids.txt")) as f:
            self.ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id_ = self.ids[idx]
        img2d = np.load(os.path.join(self.img_dir_2d, f"{id_}.npy"))[None, ...]  # [1, H, W]
        imgceus = np.load(os.path.join(self.img_dir_ceus, f"{id_}.npy"))[None, ...]
        mask = np.load(os.path.join(self.mask_dir, f"{id_}.npy"))[None, ...].astype(np.float32)
        return torch.tensor(img2d, dtype=torch.float32), torch.tensor(imgceus, dtype=torch.float32), torch.tensor(mask)

# ==== 评估指标 ====
def dice_coef(pred, target, eps=1e-6):
    num = 2 * (pred * target).sum()
    den = pred.sum() + target.sum() + eps
    return (num / den).item()

def iou_score(pred, target, eps=1e-6):
    inter = (pred * target).sum()
    union = pred.sum() + target.sum() - inter + eps
    return (inter / union).item()

def precision(pred, target, eps=1e-6):
    tp = (pred * target).sum()
    fp = (pred * (1 - target)).sum()
    return (tp / (tp + fp + eps)).item()

def recall(pred, target, eps=1e-6):
    tp = (pred * target).sum()
    fn = ((1 - pred) * target).sum()
    return (tp / (tp + fn + eps)).item()

# ==== 推理与评估 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_set = MultiModalOTUDataset(data_dir, split="test", augment=False)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

model = MultiModalSegNet().to(device)
model.load_state_dict(torch.load(checkpoint_path, map_location=device,weights_only=True))
model.eval()

dice_all, iou_all, pre_all, rec_all = [], [], [], []

with torch.no_grad():
    for i, (x2d, xceus, mask) in enumerate(test_loader):
        x2d = x2d.to(device)
        xceus = xceus.to(device)
        mask = mask.to(device)

        output = model(x2d, xceus)
        prob = torch.sigmoid(output)
        pred = (prob > 0.5).float()

        for j in range(x2d.size(0)):
            gt = mask[j, 0].cpu()
            pd = pred[j, 0].cpu()
            dice_all.append(dice_coef(pd, gt))
            iou_all.append(iou_score(pd, gt))
            pre_all.append(precision(pd, gt))
            rec_all.append(recall(pd, gt))

            idx = i * batch_size + j
            fig, axs = plt.subplots(1, 4, figsize=(12, 3))
            axs[0].imshow(x2d[j, 0].cpu(), cmap='gray')
            axs[0].set_title("2D Input")
            axs[1].imshow(xceus[j, 0].cpu(), cmap='gray')
            axs[1].set_title("CEUS Input")
            axs[2].imshow(gt, cmap='gray')
            axs[2].set_title("Ground Truth")
            axs[3].imshow(pd, cmap='gray')
            axs[3].set_title("Prediction")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f"sample_{idx}.png"))
            plt.close()

# ==== 平均指标输出 ====
print("📊 Evaluation Results on Test Set:")
print(f"Dice     : {np.mean(dice_all):.4f}")
print(f"IoU      : {np.mean(iou_all):.4f}")
print(f"Precision: {np.mean(pre_all):.4f}")
print(f"Recall   : {np.mean(rec_all):.4f}")
print(f"Sample {idx} → GT unique:", gt.unique(), "| Pred unique:", pd.unique())
