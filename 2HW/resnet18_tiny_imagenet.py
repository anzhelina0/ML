# -*- coding: utf-8 -*-
"""
ResNet18 for Tiny ImageNet (10 classes)
Clean standalone script for GitHub.
Author: <your name>
"""

# =====================
# Part 1. Imports & setup
# =====================
import os
import json
import copy
import random
import warnings
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# =====================
# Part 2. Utils
# =====================

def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_parameters(num: int) -> str:
    if num >= 1_000_000:
        return f"{num/1_000_000:.2f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)

# =====================
# Part 3. Dataset
# =====================

class TinyImageNetDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None, selected_classes=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.selected_classes = selected_classes

        with open(os.path.join(root, "wnids.txt")) as f:
            all_classes = [l.strip() for l in f]

        self.class_names = selected_classes or all_classes[:10]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.samples = []

        if split == "train":
            self._load_train()
        elif split == "val":
            self._load_val()
        elif split == "test":
            self._load_test()
        else:
            raise ValueError("Unknown split")

        print(f"{split}: {len(self.samples)} images")

    def _load_train(self):
        train_dir = os.path.join(self.root, "train")
        for cls in self.class_names:
            img_dir = os.path.join(train_dir, cls, "images")
            for img in os.listdir(img_dir):
                self.samples.append((os.path.join(img_dir, img), self.class_to_idx[cls]))

    def _load_val(self):
        ann = {}
        with open(os.path.join(self.root, "val", "val_annotations.txt")) as f:
            for line in f:
                img, cls = line.split("\t")[:2]
                if cls in self.class_to_idx:
                    ann[img] = self.class_to_idx[cls]
        img_dir = os.path.join(self.root, "val", "images")
        for img, label in ann.items():
            self.samples.append((os.path.join(img_dir, img), label))

    def _load_test(self):
        img_dir = os.path.join(self.root, "test", "images")
        for img in os.listdir(img_dir):
            self.samples.append((os.path.join(img_dir, img), -1))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# =====================
# Part 4. Model
# =====================

class BasicBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    def __init__(self, channels: List[int], blocks: List[int], num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(3, channels[0], 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels[0])
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()
        in_c = channels[0]
        for i, (out_c, n_blocks) in enumerate(zip(channels, blocks)):
            layer = []
            for j in range(n_blocks):
                stride = 2 if i > 0 and j == 0 else 1
                down = None
                if stride != 1 or in_c != out_c:
                    down = nn.Sequential(
                        nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                        nn.BatchNorm2d(out_c),
                    )
                layer.append(BasicBlock(in_c, out_c, stride, down))
                in_c = out_c
            self.layers.append(nn.Sequential(*layer))

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            x = layer(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

# =====================
# Part 5. Trainer
# =====================

class Trainer:
    def __init__(self, model):
        self.model = model.to(device)
        self.crit = nn.CrossEntropyLoss()
        self.opt = optim.Adam(model.parameters(), lr=1e-3)

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        loss_sum, correct, total = 0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = self.model(x)
            loss = self.crit(out, y)
            if train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            loss_sum += loss.item()
            correct += (out.argmax(1) == y).sum().item()
            total += y.size(0)
        return loss_sum / len(loader), 100 * correct / total

# =====================
# Part 6. Main
# =====================

def main():
    import kagglehub

    root = kagglehub.dataset_download("akash2sharma/tiny-imagenet")
    root = os.path.join(root, "tiny-imagenet-200")
    print("DATASET ROOT:", root)
    print("FILES:", os.listdir(root))



    classes = [
        'n01443537','n01629819','n01641577','n01644900','n01698640',
        'n03089624','n01770393','n01774750','n03976657','n02395406'
    ]

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    val_tf = transforms.Compose([
        transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_ds = TinyImageNetDataset(root, "train", train_tf, classes)
    val_ds = TinyImageNetDataset(root, "val", val_tf, classes)

    train_ld = DataLoader(train_ds, 32, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, 32, shuffle=False, num_workers=2)

    model = ResNet18([64,128,256], [2,2,2])
    print("Params:", format_parameters(count_parameters(model)))

    trainer = Trainer(model)

    history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
    }

    for epoch in range(20):
        tr_l, tr_a = trainer.run_epoch(train_ld, True)
        va_l, va_a = trainer.run_epoch(val_ld, False)

        history["train_loss"].append(tr_l)
        history["train_acc"].append(tr_a)
        history["val_loss"].append(va_l)
        history["val_acc"].append(va_a)

        print(f"Epoch {epoch+1}: train acc={tr_a:.2f}%, val acc={va_a:.2f}%")

    # ---------- plots ----------
    import matplotlib.pyplot as plt
    os.makedirs("plots", exist_ok=True)

    epochs = range(1, len(history["train_acc"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="Train Acc")
    plt.plot(epochs, history["val_acc"], label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.savefig("plots/accuracy.png")
    plt.close()

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("plots/loss.png")
    plt.close()

    # ---------- table ----------
    results = pd.DataFrame({
        "Stage": ["Baseline"],
        "Configuration": ["ResNet18 64→128→256"],
        "Parameters": [format_parameters(count_parameters(model))],
        "Train Accuracy": [f"{history['train_acc'][-1]:.2f}%"],
        "Validation Accuracy": [f"{history['val_acc'][-1]:.2f}%"]
    })

    results.to_csv("results_baseline.csv", index=False)

    with open("results.json", "w") as f:
        json.dump(history, f, indent=2)



if __name__ == "__main__":
    main()
