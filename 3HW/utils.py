import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

class MoonDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir/
          images/
            render/  - входные картинки
            ground/  - бинарные маски
        """
        self.img_dir = os.path.join(root_dir, "images", "render")
        self.mask_dir = os.path.join(root_dir, "images", "ground")

        self.img_names = sorted(os.listdir(self.img_dir))
        self.mask_names = sorted(os.listdir(self.mask_dir))
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

        # бинаризация маски
        mask = (mask > 0).float()

        return image, mask


def get_transforms():
    return T.Compose([
        T.Resize((128, 128)),
        T.ToTensor(),
    ])
import torch

def dice_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))

    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


def iou_score(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    intersection = (pred * target).sum(dim=(1,2,3))
    union = pred.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()


def pixel_accuracy(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()

    correct = (pred == target).float().sum()
    total = target.numel()

    return (correct / total).item()

