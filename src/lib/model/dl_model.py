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

        
    def features(self, image):
        '''
        Extract feature of images
        '''
        return self.model(image)
        
    def forward(self, image):
        return self.model(image)