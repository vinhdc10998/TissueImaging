import torch
import cv2
import os
import pandas as pd
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

class UltraSoundDataset(Dataset):
    def __init__(self, root="./data/Dataset_BUSI_with_GT", transform=None, classes = None):
        # Initialize dataset paths and parameters
        self.data = pd.read_csv(root)
        self.weight_1 = self.data['weight_1']
        self.weight_2 = self.data['weight_2']
        self.weight_3 = self.data['weight_3']
        self.transform = transform
        self.categories = classes
        self.labels = self.data['label']
        
        # # Collect image names and labels for each category
        # for idx, cat in enumerate(self.categories):
        #     cat_path = os.path.join(self.root, cat)
        #     files = [f.replace(".png", "") for f in os.listdir(cat_path) if "mask" not in f]
        #     self.img_names.extend(files)
        #     self.labels.extend([idx] * len(files))
            
    def __len__(self):
        # Return dataset size
        return len(self.data)
    
    def __getitem__(self, index):
        # Load image and mask

        # img_name, label = self.img_names[item], self.labels[item]
        img_wei1, img_wei2, img_wei3, label = self.weight_1[index], self.weight_2[index], self.weight_3[index], self.labels[index]

        if img_wei1 is not np.nan:
            img1 = cv2.imread(img_wei1, cv2.IMREAD_COLOR)
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            if self.transform:
                img1 = self.transform(img1)
        else: img1 = torch.zeros((3, 224, 224))

        if img_wei2 is not np.nan:
            img2 = cv2.imread(img_wei2, cv2.IMREAD_COLOR)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            if self.transform:
                img2 = self.transform(img2)
        else: img2 = torch.zeros((3, 224, 224))


        if img_wei3 is not np.nan:
            img3 = cv2.imread(img_wei3, cv2.IMREAD_COLOR)
            img3 = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
            if self.transform:
                img3 = self.transform(img3)
        else: img3 = torch.zeros((3, 224, 224))


        # apply transformations
        return {
            'weight_1': img1,
            'weight_2': img2,
            'weight_3': img3,
            'label': label
        }
