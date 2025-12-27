import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from models.unet_with_backbone import UNetWithBackbone
from utils import MoonDataset, get_transforms, dice_score, iou_score, pixel_accuracy

# --------------------
# Параметры
# --------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
data_root = "data"
batch_size = 4
epochs = 5
lr = 1e-3
val_ratio = 0.2

# --------------------
# Dataset
# --------------------
transform = get_transforms()
dataset = MoonDataset(data_root, transform=transform)

val_size = int(len(dataset) * val_ratio)
train_size = len(dataset) - val_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size)

# --------------------
# Модель
# --------------------
model = UNetWithBackbone().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# --------------------
# Обучение
# --------------------
for epoch in range(epochs):
    model.train()
    train_loss = 0

    for imgs, masks in train_loader:
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    from metrics import dice_coef, iou_score, pixel_accuracy

    model.eval()
    dice, iou, acc = 0, 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            preds = torch.sigmoid(model(x))
            dice += dice_coef(preds, y)
            iou += iou_score(preds, y)
            acc += pixel_accuracy(preds, y)

    dice /= len(val_loader)
    iou /= len(val_loader)
    acc /= len(val_loader)

    print(f"Val Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {acc:.4f}")


    # --------------------
    # Валидация
    # --------------------
    model.eval()
    dice, iou, acc = 0, 0, 0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            outputs = model(imgs)

            dice += dice_score(outputs, masks)
            iou += iou_score(outputs, masks)
            acc += pixel_accuracy(outputs, masks)

    dice /= len(val_loader)
    iou /= len(val_loader)
    acc /= len(val_loader)

    print(
        f"Epoch [{epoch+1}/{epochs}] "
        f"Loss: {train_loss/len(train_loader):.4f} | "
        f"Dice: {dice:.4f} | IoU: {iou:.4f} | Acc: {acc:.4f}"
    )

# --------------------
# Сохранение модели
# --------------------
torch.save(model.state_dict(), "unet_moon.pth")
print("Model saved as unet_moon.pth")
