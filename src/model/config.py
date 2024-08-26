import os

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, 
                                ElasticNet, MultiTaskElasticNet)
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.ensemble import (AdaBoostRegressor, ExtraTreesRegressor, 
                            GradientBoostingRegressor, RandomForestRegressor)
from lightgbm import LGBMRegressor
from utils.ultrasoundDataset import UltraSoundDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset

class Config:
    def __init__(self, images, device, epochs=20, type_model='mlmodel', lr=5e-4, weight_decay=1e-5, test_size=0):
        self.ML_models = self.__get_ML_models()
        self.data = self.get_data(images, test_size)
        self.epochs = epochs
        self.type_model = type_model
        self.device = device
        self.lr = lr
        self.weight_decay = weight_decay
        self.log_path = self.get_log_path(self.type_model)
    
    @staticmethod
    def get_data(images, test_size):
        """ 
            Get images path
            Return: UltraSoundDataset 
        """
        USdataset = UltraSoundDataset(images)
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
        
    @staticmethod
    def __get_ML_models():
        return [
            (LinearRegression(), 'Linear Regression'),
            (Ridge(alpha=1.0), 'Regularization L2 Linear Regression'),
            (Lasso(alpha=0.05), 'Regularization L1 Linear Regression'),
            (ElasticNet(), 'ElasticNet'),
            (MultiTaskElasticNet(alpha=0.1), 'MultiTask ElasticNet'),
            (SVR(C=1.0, epsilon=0.2), 'Support Vector Regression'),
            (DecisionTreeRegressor(), 'Decision Tree Regression'),
            (xgb.XGBRegressor(verbosity=0), 'XGBoost'),
            (AdaBoostRegressor(random_state=0, n_estimators=100), 'AdaBoost'),
            (ExtraTreesRegressor(), 'ExtraTreeRegressor'),
            (GradientBoostingRegressor(), 'GradientBoostingRegressor'),
            (RandomForestRegressor(max_depth=2, random_state=0), 'RandomForest'),
            (LGBMRegressor(), 'LightGBM')
        ]

    def __getitem__(self, item):
        return self.config[item]

    def __contains__(self, item):
        return item in self.config