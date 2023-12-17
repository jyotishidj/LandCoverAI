import os
from zipfile import ZipFile
import torch
from torch import nn
from pathlib import Path
from LCover.entity.config_entity import PrepareBaseModelConfig
from LCover.utils.common import UNet


device = "cuda" if torch.cuda.is_available() else "cpu"

class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = UNet(out_channels = self.config.params_classes).to(device)

        self.save_model(path=self.config.updated_base_model_path, model=self.model)  # Path will be set to "self.config.base_model_path", if pretrained model downloaded 
    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        pass
    
    
    def update_base_model(self):
        pass


    @staticmethod
    def save_model(path: Path, model):
        torch.save({'model_state_dict': model.state_dict()}, path)

