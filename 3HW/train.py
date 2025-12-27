import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from models.unet import UNet
from utils import MoonDataset, get_transforms

# -------------------
# Параметры
# -------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
img_dir = "data/images"
mask_dir = "data/masks"
batch_size = 4
epochs = 10
lr = 1e-3

# -------------------
# Датасет и загрузчик
# -------------------
transform = get_transforms()
dataset = MoonDataset(img_dir, mask_dir, transform=transform)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# -------------------
# Модель, лосс, оптимайзер
# -------------------
model = UNet().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# -------------------
# Тренировка
# -------------------
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}")

# -------------------
# Сохраняем модель
# -------------------
torch.save(model.state_dict(), "unet_moon.pth")
print("Model saved as unet_moon.pth")
