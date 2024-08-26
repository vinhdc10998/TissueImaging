import torch
import numpy as np
import logging


class Runner():
    def __init__(self, cfg):
        self.cfg = cfg
        self.setup_logging()

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

    def run(self):
        if self.cfg.type_model == 'mlmodel':
            self.logger.info("RUN MACHINE LEARNING MODELS")
            for model, name in self.cfg.ML_models:
                    self.logger.info(f'Model: {name}')