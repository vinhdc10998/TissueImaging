import torch
import cv2
import os

import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

class UltraSoundDataset(Dataset):
    def __init__(self, root="./data/Dataset_BUSI_with_GT", transform=None):
        # Initialize dataset paths and parameters
        self.root = root
        self.img_names = []
        self.transform = transform
        self.categories = ['normal', 'benign', 'malignant']
        self.labels = []
        
        # Collect image names and labels for each category
        for idx, cat in enumerate(self.categories):
            cat_path = os.path.join(self.root, cat)
            files = [f.replace(".png", "") for f in os.listdir(cat_path) if "mask" not in f]
            self.img_names.extend(files)
            self.labels.extend([idx] * len(files))
            
    def __len__(self):
        # Return dataset size
        return len(self.img_names)
    
    def __getitem__(self, item):
        # Load image and mask
        img_name, label = self.img_names[item], self.labels[item]
        img = cv2.imread(os.path.join(self.root, self.categories[label], img_name + ".png"), cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # apply transformations
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
        
        return img, label
