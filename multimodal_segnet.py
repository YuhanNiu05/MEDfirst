
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# === 基础模块 ===
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            ConvBlock(in_c, out_c)
        )
    def forward(self, x):
        return self.pool_conv(x)

class Up(nn.Module):
    def __init__(self, in_c, skip_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, skip_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(skip_c * 2, out_c)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# === 共享/特异特征融合模块 ===
class SharedDistinctFusion(nn.Module):
    def __init__(self, c2d, cceus):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Conv2d(c2d + cceus, c2d, 1),
            nn.ReLU(),
            nn.Conv2d(c2d, c2d, 1)
        )
        self.distinct_2d = nn.Sequential(
            nn.Conv2d(c2d, c2d, 1),
            nn.Sigmoid()
        )
        self.distinct_ceus = nn.Sequential(
            nn.Conv2d(cceus, cceus, 1),
            nn.Sigmoid()
        )

    def forward(self, f2d, fceus):
        shared = self.shared(torch.cat([f2d, fceus], dim=1))
        d2d = self.distinct_2d(f2d) * f2d
        dceus = self.distinct_ceus(fceus) * fceus
        return torch.cat([shared, d2d, dceus], dim=1)  # 通道数: 64 + 64 + 64 = 192

# === CEUS 编码器 ===
class CEUSEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT
        base = mobilenet_v3_small(weights=weights)
        base.features[0][0] = nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1, bias=False)
        self.features = nn.Sequential(*list(base.features.children())[:6])
        self.project = nn.Conv2d(40, 64, 1)

    def forward(self, x):
        x = self.features(x)
        return self.project(x)

# === 主模型 ===
class MultiModalSegNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.inc2d = ConvBlock(in_ch, 64)
        self.down1_2d = Down(64, 64)
        self.enc_ceus = CEUSEncoder()
        self.fusion = SharedDistinctFusion(c2d=64, cceus=64)
        self.up1 = Up(in_c=192, skip_c=64, out_c=64)  # ✅ 通道数严格对齐
        self.outc = nn.Conv2d(64, out_ch, 1)

    def forward(self, x2d, xceus):
        f2d = self.down1_2d(self.inc2d(x2d))          # [B, 64, 128, 128]
        fceus = self.enc_ceus(xceus)                  # [B, 64, ~64, ~64]
        fceus_up = F.interpolate(fceus, size=f2d.shape[2:], mode='bilinear', align_corners=False)
        fcat = self.fusion(f2d, fceus_up)             # [B, 192, 128, 128]
        x = self.up1(fcat, self.inc2d(x2d))           # skip: 2D输入特征
        return self.outc(x)
