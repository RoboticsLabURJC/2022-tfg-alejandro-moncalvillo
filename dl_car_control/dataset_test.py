import os
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision import transforms
from torchvision.io import read_image,ImageReadMode
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2
from PIL import Image


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path)
        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        img_tensor = transform(image)
        label = torch.FloatTensor((self.img_labels.iloc[idx, 1], self.img_labels.iloc[idx, 2]))
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = torch.FloatTensor(label)
        return img_tensor, label



def main():
    training_data = CustomImageDataset('output.csv','dataset')
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    train_features, train_labels = next(iter(train_dataloader))
    #torch.Size([3, 239, 640])
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0]
    label = train_labels[0]
    
    print(img.type())
    print(img.size())


    



    

# Execute!
if __name__ == "__main__":
    main()