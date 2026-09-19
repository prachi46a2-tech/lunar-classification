import os
import math
import argparse
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

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

class SunFiLMNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.resnet18(weights=None)
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

class LunarTestDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

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
        return image, sun_vec, str(row['image_id'])

def run_inference(weights_path, test_csv, test_dir, output_csv, threshold=0.8329):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SunFiLMNet().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    df = pd.read_csv(test_csv)
    dataset = LunarTestDataset(df, test_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_ids, all_probs = [], []
    with torch.no_grad():
        for imgs, vecs, ids in loader:
            imgs, vecs = imgs.to(device), vecs.to(device)
            probs = torch.sigmoid(model(imgs, vecs)).cpu().tolist()
            all_probs.extend(probs)
            all_ids.extend(ids)

    preds = [1 if p >= threshold else 0 for p in all_probs]
    clean_ids = [str(i).replace('.png', '') for i in all_ids]
    
    sub = pd.DataFrame({'image_id': clean_ids, 'label': preds})
    sub.to_csv(output_csv, index=False)
    print(f"Saved {len(sub)} predictions to {output_csv} using threshold {threshold}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default='final_lunar_model.pth')
    parser.add_argument('--test_csv', default='test.csv')
    parser.add_argument('--test_dir', default='test_images')
    parser.add_argument('--output', default='submission.csv')
    parser.add_argument('--threshold', type=float, default=0.8329)
    args = parser.parse_args()
    
    run_inference(args.weights, args.test_csv, args.test_dir, args.output, args.threshold)
