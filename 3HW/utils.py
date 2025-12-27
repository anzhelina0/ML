import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MoonDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted(os.listdir(img_dir))
        self.mask_names = sorted(os.listdir(mask_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # нормализуем маску в 0/1
        mask = torch.where(mask>0, 1.0, 0.0)
        return image, mask

def get_transforms():
    return T.Compose([
        T.Resize((128,128)),
        T.ToTensor(),
    ])
