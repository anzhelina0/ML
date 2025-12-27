import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Блок двойной свёртки
# ---------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

# ---------------------------
# Up блок с транспонированной свёрткой и конкатенацией
# ---------------------------
class Up(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x

# ---------------------------
# Полная U-Net архитектура
# ---------------------------
class UNet(nn.Module):
    def __init__(self, in_channels=3, base_c=32):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, base_c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c, base_c*2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c*2, base_c*4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c*4, base_c*8))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(base_c*8, base_c*16))

        self.up1 = Up(base_c*16, base_c*8, base_c*8)
        self.up2 = Up(base_c*8, base_c*4, base_c*4)
        self.up3 = Up(base_c*4, base_c*2, base_c*2)
        self.up4 = Up(base_c*2, base_c, base_c)

        self.outc = nn.Conv2d(base_c, 1, kernel_size=1)  # 1 канал для бинарной маски

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x

# ---------------------------
# Тестирование модели
# ---------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    x = torch.randn(1, 3, 128, 128).to(device)
    y = model(x)
    print("Output shape:", y.shape)
