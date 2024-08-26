import torch
import argparse
import albumentations as A

from model.config import Config
from model.runner import Runner
from utils.ultrasoundDataset import UltraSoundDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['test'] = Subset(dataset, val_idx)
    return datasets

def parse_args():
    parser = argparse.ArgumentParser(description="Tissue Imaging")
    parser.add_argument("model", choices=["mlmodel", "dlmodel", "all"], 
                        help="ML Model, DL Model or Run all Models?")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate (Default: 5e-4)")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight_Decay (Default: 1e-5)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to test the model on (Default: 50)")
    args = parser.parse_args()
    return args



def train():
    # Init parameter 
    args = parse_args()
    device = torch.device('cpu') if not torch.cuda.is_available() or args.cpu else torch.device('cuda')
    cfg = Config(device, args.one_hot, args.drop_col, args.pca_transform, args.mode,
                    args.epochs, args.model, args.kfold, args.lr, args.weight_decay)

    data_path = './data/Dataset_BUSI_with_GT'

    # Load dataset
    ultraSoundDataset = UltraSoundDataset(data_path)
    ultraSoundDataset = train_val_dataset(ultraSoundDataset, 0.2)
    print(len(ultraSoundDataset['train']))
    print(len(ultraSoundDataset['test']))

    # model


if __name__ == '__main__':
    train()