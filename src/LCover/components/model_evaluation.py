import torch
import os
from pathlib import Path
import mlflow
import mlflow.keras
from urllib.parse import urlparse
from LCover.entity.config_entity import EvaluationConfig
from LCover.utils.common import UNet, SegmentationDataset, save_json, class_report
from torch.utils.data import DataLoader
import torchmetrics
import torch.nn.functional as F
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

class Evaluation:
    def __init__(self, config: EvaluationConfig):
        self.config = config

    
    def test_generator(self):

        test_set = SegmentationDataset(DATA_DIR=self.config.root_data_dir,mode='test',ratio=1,transforms=None)
        self.test_loader = DataLoader(test_set, batch_size = self.config.params_batch_size, shuffle = True,  num_workers = 1)
    

    def evaluation(self):
        
        stat_scores = torchmetrics.StatScores(task = 'multiclass', num_classes = self.config.params_classes, average = None, multidim_average = "global").to(device)
        acc = torchmetrics.Accuracy(task = 'multiclass', num_classes = self.config.params_classes, average = "micro", multidim_average = "global").to(device)
        jaccard = torchmetrics.JaccardIndex(task = 'multiclass', num_classes = self.config.params_classes,).to(device)
        
        self.model = self.load_model()
        self.model.eval()


        class_probs = {0: 0, 1: 0, 2: 0, 3: 0, 4:0}
        num_samples = {0:0, 1:0, 2:0, 3:0, 4:0}

        for X,y in self.test_loader:
            X = X.to(device)
            y = y.to(device)

            with torch.no_grad():
                logits = F.softmax(self.model(X), dim =1)
                aggr = torch.max(logits, dim = 1)
                #preds = aggr[1].cpu().numpy().flatten()
                preds = aggr[1]
                probs = aggr[0]
                for label in class_probs.keys():
                    class_probs[label]+= probs[preds == label].flatten().sum()
                    num_samples[label]+= preds[preds == label].flatten().size(dim = 0)
                #predictions_list = np.concatenate((predictions_list, preds))
                stat_scores.update(preds, y)
                acc.update(preds,y)
                jaccard.update(preds, y)
        for label in class_probs.keys():
            class_probs[label] /= num_samples[label]

        target_names = np.array(["background", "building", "woodland", "water", "road"])

        self.score = class_report(target_names, stat_scores.compute(), acc.compute(), 
                     jaccard.compute(),class_probs) #Acc,Jaccard,F1
        
        self.save_score()
        self.stage_model()
        self.log_into_mlflow()

 
    def save_score(self):
        scores = {"Accuracy": self.score[0], "Jaccard": self.score[1], "F1": self.score[2]}
        save_json(path=Path(f"scores_{self.config.model_name}_{self.config.model_version}.json"), data=scores)

    
    def log_into_mlflow(self):


        with mlflow.start_run():
            mlflow.log_params(self.config.all_params)
            mlflow.log_metrics(
                {"Accuracy": self.score[0], "Jaccard": self.score[1], "F1": self.score[2]}
            )
            mlflow.pytorch.log_model(self.model, "model")
            mlflow.pytorch.log_state_dict(self.model.state_dict(), artifact_path="model")

    
    def load_model(self):
        model = UNet(out_channels = self.config.params_classes).to(device)
        checkpoint = torch.load(
            self.config.path_of_model,map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    def stage_model(self):
        os.makedirs(self.config.stage_model, exist_ok=True)
        torch.save({'model_state_dict': self.model.state_dict()}, os.path.join(self.config.stage_model
                                            ,f"scores_{self.config.model_name}_{self.config.model_version}.pth"))
        
