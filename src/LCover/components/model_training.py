import os
import time
import torch
from torch import nn
from pathlib import Path
from LCover.entity.config_entity import TrainingConfig
from LCover.entity.config_entity import PrepareBaseModelConfig
from LCover.utils.common import UNet, SegmentationDataset, EarlyStopping
import albumentations as A
from torch.utils.data import DataLoader
from torch.optim import Adam
import segmentation_models_pytorch as smp
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config

    
    def get_base_model(self):
        self.model = UNet(out_channels = self.config.params_classes).to(device)
        checkpoint = torch.load(
            self.config.updated_base_model_path,map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

    # Data Loader
    def train_valid_generator(self): 
        
        if self.config.params_is_augmentation:
            transforms = A.Compose([
                            A.OneOf([
                                A.HueSaturationValue(40,40,30,p=1),
                                A.RandomBrightnessContrast(p=1,brightness_limit = 0.2,
                                                        contrast_limit = 0.5)], p = 0.5),
                            A.OneOf([
                                A.RandomRotate90(p=1),
                                A.HorizontalFlip(p=1),
                                A.RandomSizedCrop(min_max_height=(248,512),height=512,width=512, p =1)
                            ], p = 0.5)])
        else:
            transforms = None

        train_set = SegmentationDataset(DATA_DIR=self.config.root_data_dir,mode='train',ratio=self.config.params_ratio,transforms=transforms)
        self.train_loader = DataLoader(train_set, batch_size = self.config.params_batch_size, shuffle = True,  num_workers = 1)

        valid_set = SegmentationDataset(DATA_DIR=self.config.root_data_dir,mode='val',ratio=1,transforms=None)
        self.valid_loader = DataLoader(valid_set, batch_size = self.config.params_batch_size, shuffle = True,  num_workers = 1)


    def train(self):

        reg_lambda = 1e-6
        mod_epochs=5
        patience = 7
        loss_fn = smp.losses.JaccardLoss(mode = "multiclass", classes = self.config.params_classes).to(device)
        optim = Adam(self.model.parameters(), lr=self.config.params_learning_rate)

        train_loss_list = []
        val_loss_list = []
        num_train_batches = len(self.train_loader)
        num_valid_batches = len(self.valid_loader)
        counter_epochs = 0
        min_valid_loss = np.Inf

        if self.config.parmas_early_stopping:
            ear_stopping = EarlyStopping(patience= patience)

        tic = time.time()
        for epoch in range(self.config.params_epochs):
            counter_epochs+=1
            self.model.train()
            train_loss, valid_loss = 0.0, 0.0
            for train_batch in self.train_loader:
                X, y = train_batch[0].to(device), train_batch[1].to(device)
                preds = self.model(X)

                loss = loss_fn(preds, y)
                train_loss += loss.item()
            
                l_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                loss = loss + reg_lambda * l_norm

                # Backpropagation
                optim.zero_grad()
                loss.backward()
                optim.step()

                
            self.model.eval()
            with torch.no_grad():
                for val_batch in self.valid_loader:
                    X, y = val_batch[0].to(device), val_batch[1].to(device)
                    preds = self.model(X)
                    
                    valid_loss += loss_fn(preds, y).item()
                    
            train_loss /= num_train_batches
            valid_loss /= num_valid_batches

            train_loss_list.append(train_loss)
            val_loss_list.append(valid_loss)

            if (epoch + 1) % mod_epochs == 0:
                print(f"Epoch: {epoch + 1}/{self.config.params_epochs}{5 * ' '}Training Loss: {train_loss:.4f}{5 * ' '}Validation Loss: {valid_loss:.4f}")

            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                self.save_model(path=self.config.trained_model_path,
                                model=self.model)

            if self.config.parmas_early_stopping:
                ear_stopping(valid_loss, self.model)
                if ear_stopping.early_stop:
                    print("Early stopping")
                    break


        total_time = time.time() - tic
        mins, secs = divmod(total_time, 60)
        if mins < 60:
            print(f"\n Training completed in {mins} m {secs:.2f} s.")
        else:
            hours, mins = divmod(mins, 60)
            print(f"\n Training completed in {hours} h {mins} m {secs:.2f} s.")
        
    @staticmethod
    def save_model(path: Path, model):
        torch.save({'model_state_dict': model.state_dict()}, path)

    

