from torch import nn
import torch
import timm

class DeepLearningModel(nn.Module):
    def __init__(self, cfg, inp_channels=3):
        super().__init__()
        self.model = timm.create_model(
            cfg.model, 
            pretrained=False, 
            num_classes=len(cfg.classes), 
            in_chans=inp_channels
        )
        self.softmax = nn.Softmax(dim=1)
        # n_features = self.model.head.in_features
        # self.model.head = nn.Linear(n_features, 256)
        # self.fc = nn.Sequential(
        #     nn.Linear(256, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, len(cfg.classes))
        # )
        # self.dropout = nn.Dropout(dropout)
        
    def features(self, image):
        '''
        Extract feature of images
        '''
        return self.model(image)
        
    def forward(self, image):
        # embeddings = self.features(image)
        # x = self.dropout(embeddings)
        # output = self.fc(x)
        return self.model(image)