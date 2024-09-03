import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from .model.dl_model import DeepLearningModel
from sklearn.metrics import f1_score, recall_score, precision_score

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_logging()
        self.model = DeepLearningModel(cfg).to(cfg.device)
        self.loss = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size
        self.metrics = {
            'recall': 0,
            'f1_score': 0,
            'precision': 0

        }

    def setup_logging(self):
        formatter = logging.Formatter("[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(self.cfg.log_path)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)
        logging.basicConfig(level=logging.DEBUG, handlers=[file_handler, stream_handler])
        self.logger = logging.getLogger(__name__)

    def calc_metrics(self, predicted, y):
        self.metrics['recall'] = recall_score(y, predicted)
        self.metrics['f1_score'] = f1_score(y, predicted)
        self.metrics['precision'] = precision_score(y, predicted)

    def train(self):
        self.logger.info("RUN DEEP LEARNING MODELS")
        self.logger.debug(f"DEVICE-IN-USE: {self.cfg.device}")
        self.logger.info(f"USE DEVICE {self.cfg.device}")
        train_loader = DataLoader(self.cfg.data['train'], batch_size=self.batch_size, shuffle=False, pin_memory=True)
        start_epochs = 1
        self.model.train()
        for epoch in range(start_epochs, self.epochs + 1):
            # self.logger.info(f"EPOCHS {epoch}")
            running_loss = 0.0
            predictions = []
            labels = []
            for batch, (X, y) in enumerate(train_loader):
                    X, y = X.to(self.cfg.device), y.to(self.cfg.device)
                    self.optimizer.zero_grad()
                    y_hat = self.model(X)
                    loss = self.loss(y_hat, y)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()

                    predictions.append(torch.argmax(y_hat, dim=1))
                    labels.append(y)

            predictions = torch.cat(predictions, dim=0).cpu().detach().numpy()
            labels = torch.cat(labels, dim=0).cpu().detach().numpy()
            self.calc_metrics(predictions, labels)
            self.logger.info(f'[Epoch {epoch}] loss: {running_loss:.3f}\trecall: {self.metrics["recall"]}\tprecision: {self.metrics["precision"]}\tF1-score: {self.metrics["f1_score"]}')
        self.logger.info("FINISHED TRAINING")