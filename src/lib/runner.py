import torch
import numpy as np
import logging
from torch.utils.data import DataLoader
from .model.dl_model import DeepLearningModel

class Runner:
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_logging()
        self.model = DeepLearningModel(cfg)
        self.loss = torch.nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), cfg.lr)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.995)
        self.epochs = cfg.epochs
        self.batch_size = cfg.batch_size

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

    def train(self):
        self.logger.info("RUN DEEP LEARNING MODELS")
        self.logger.debug(f"DEVICE-IN-USE: {self.cfg.device}")

        train_loader = DataLoader(self.cfg.data['train'], batch_size=self.batch_size, shuffle=False, pin_memory=True)
        start_epochs = 1
        self.model.train()
        for epoch in range(start_epochs, self.epochs + 1):
            self.logger.info(f"EPOCHS {epoch}")
            running_loss = 0.0

            for batch, (X, y) in enumerate(train_loader):
                    X, y = X.to(self.cfg.device), y.to(self.cfg.device)
                    self.optimizer.zero_grad()
                    prediction = self.model(X)
                    loss = self.loss(prediction, y)
                    loss.backward()
                    self.optimizer.step()
                    running_loss += loss.item()
                    if batch % 2000 == 1999:    # print every 2000 mini-batches
                        print(f'[{epoch + 1}, {batch + 1:5d}] loss: {running_loss / 2000:.3f}')
                        running_loss = 0.0
        self.logger.info("FINISHED TRAINING")

        # X_train, X_val, y_train, y_val = [torch.from_numpy(data).float().to(self.cfg.device) for data in self.cfg.data]
            # assert len(X_train) == len(y_train) and len(X_val) == len(y_val)