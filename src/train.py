import torch
import argparse
import albumentations as A

from utils.ultrasoundDataset import UltraSoundDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets

def train():
    # Init parameter 
    data_path = './data/Dataset_BUSI_with_GT'

    # Load dataset
    ultraSoundDataset = UltraSoundDataset(data_path)
    ultraSoundDataset = train_val_dataset(ultraSoundDataset, 0.2)
    print(len(ultraSoundDataset['train']))
    print(len(ultraSoundDataset['test']))

    # model


if __name__ == '__main__':
    train()