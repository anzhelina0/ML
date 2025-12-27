import torch
import matplotlib.pyplot as plt
from models.unet import UNet
from utils import MoonDataset, get_transforms

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = MoonDataset("data", transform=get_transforms())
model = UNet().to(device)
model.load_state_dict(torch.load("unet_moon.pth", map_location=device))
model.eval()

for i in range(3):  # 3–5 картинок
    img, mask = dataset[i]
    img = img.unsqueeze(0).to(device)

    with torch.no_grad():
        pred = torch.sigmoid(model(img))[0,0].cpu()

    plt.figure(figsize=(9,3))

    plt.subplot(1,3,1)
    plt.title("Input")
    plt.imshow(img[0].permute(1,2,0).cpu())
    plt.axis("off")

    plt.subplot(1,3,2)
    plt.title("Prediction")
    plt.imshow(pred > 0.5, cmap="gray")
    plt.axis("off")

    plt.subplot(1,3,3)
    plt.title("GT")
    plt.imshow(mask[0], cmap="gray")
    plt.axis("off")

    plt.show()
