import os
import torchvision.transforms as transforms

from utils.ultrasoundDataset import UltraSoundDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class Config:
    def __init__(self, data, device, epochs=20, type_model='mlmodel', lr=5e-4, batch_size=32, test_size=0):
        self.epochs = epochs
        self.type_model = type_model
        self.device = device
        self.lr = lr
        self.batch_size = batch_size
        self.log_path = self.get_log_path(self.type_model)
        self.classes = ['normal', 'benign']
        self.model = 'resnet50'
        self.data = self.get_data(data, test_size, self.classes)

    @staticmethod
    def get_data(data, test_size, classes):
        """ 
            Get images path
            Return: UltraSoundDataset 
        """
        img_size = (224, 224)

        transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Resize(img_size),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        USdataset = UltraSoundDataset(data, transform, classes)
        datasets = {}

        if test_size == 0:
            datasets['train'] = USdataset
            datasets['test'] = []
        else:
            train_idx, val_idx = train_test_split(list(range(len(USdataset))), test_size=test_size)
            datasets['train'] = Subset(USdataset, train_idx)
            datasets['test'] = Subset(USdataset, val_idx)
        return datasets

    @staticmethod
    def get_log_path(type_model):
        logging_path = os.path.join(os.getcwd(), 'logging')
        if not os.path.exists(logging_path):
            os.mkdir(logging_path)
        return os.path.join(logging_path, 'log_{}.txt'.format(type_model))

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config