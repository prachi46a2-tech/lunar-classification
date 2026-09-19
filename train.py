import os
import math
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# --- Dataset Definition ---
class LunarDataset(Dataset):
    def __init__(self, df, img_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row['image_id'])
        if not img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert('L')
        if self.transform:
            image = self.transform(image)
            
        angle = float(row['sun_azimuth_angle'])
        rad = math.radians(angle)
        sun_vec = torch.tensor([math.sin(rad), math.cos(rad)], dtype=torch.float32)
        
        if not self.is_test:
            label = torch.tensor(float(row['label']), dtype=torch.float32)
            return image, sun_vec, label
        return image, sun_vec, str(row['image_id'])

# --- FiLM Block ---
class FiLMBlock(nn.Module):
    def __init__(self, in_features=2, num_channels=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_channels * 2)
        )
        self.num_channels = num_channels

    def forward(self, x, sun_vec):
        params = self.mlp(sun_vec)
        gamma = params[:, :self.num_channels].unsqueeze(2).unsqueeze(3)
        beta = params[:, self.num_channels:].unsqueeze(2).unsqueeze(3)
        return gamma * x + beta

# --- SunFiLMNet Model ---
class SunFiLMNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool

        self.layer1 = base.layer1
        self.film1 = FiLMBlock(2, 64)
        self.layer2 = base.layer2
        self.film2 = FiLMBlock(2, 128)
        self.layer3 = base.layer3
        self.film3 = FiLMBlock(2, 256)
        self.layer4 = base.layer4
        self.film4 = FiLMBlock(2, 512)

        self.avgpool = base.avgpool
        self.fc = nn.Linear(512, 1)

    def forward(self, x, sun_vec):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.film1(self.layer1(x), sun_vec)
        x = self.film2(self.layer2(x), sun_vec)
        x = self.film3(self.layer3(x), sun_vec)
        x = self.film4(self.layer4(x), sun_vec)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x).squeeze(1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on: {device}")
    model = SunFiLMNet().to(device)
    torch.save(model.state_dict(), "final_lunar_model.pth")
    print("Model initialized and template weights checkpoint saved.")

if __name__ == '__main__':
    main()
