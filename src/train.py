import torch
import argparse
import albumentations as A
import logging

from lib.config import Config
from lib.runner import Runner

def parse_args():
    parser = argparse.ArgumentParser(description="Tissue Imaging")
    parser.add_argument("model", choices=["mlmodel", "dlmodel", "all"], 
                        help="ML Model, DL Model or Run all Models?")
    parser.add_argument("--images", type=str, default='./data/Dataset_BUSI_with_GT', help="path of tissue image")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning Rate (Default: 5e-4)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch Size (Default: 32)")
    parser.add_argument("--epochs", type=int, default=50, help="Epochs to test the model on (Default: 50)")
    parser.add_argument("--gpu", action="store_true", default=False, required=False, help="Use GPU (Default: CPU)")

    args = parser.parse_args()
    return args



def train():
    # Init parameter 
    args = parse_args()
    device = torch.device('cpu') if not torch.cuda.is_available() or not args.gpu else torch.device('cuda')
    cfg = Config(args.images, device, args.epochs, args.model, args.lr, args.batch_size, test_size=0.2)

    if args.model in ['all', 'mlmodel', 'dlmodel']:
        try:
            runner = Runner(cfg)
            runner.train()
        except KeyboardInterrupt:
            logging.info('Building Model interrupted.')
    return


if __name__ == '__main__':
    train()