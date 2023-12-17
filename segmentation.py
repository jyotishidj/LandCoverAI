
import numpy as np
import torch
import torch.nn as nn
import os
from src.LCover.utils.common import UNet
import cv2

device = "cuda" if torch.cuda.is_available() else "cpu"

class SegmentationPipeline:
    def __init__(self, model_name, model_version):
        self.model_name = model_name
        self.model_version = model_version

    
    def segment(self,image):
        # load model
        model = self.load_model(model_name=self.model_name,model_version=self.model_version)
        model.eval()
 
        img, meta_info = self.split_image(image)
    
        pred_mask = []
        for i in range(img.shape[0]):
            pred =torch.argmax(model(img[i][None,:,:,:].to(device)), dim = 1)
            pred_mask.append(torch.squeeze(pred).detach().cpu().numpy())

        pred_mask = np.stack(pred_mask,axis=0)
        mask = self.join_image(pred_mask, meta_info)

        return mask
        
    @staticmethod
    def load_model(model_name, model_version):
        model = UNet(out_channels = 5).to(device)
        checkpoint = torch.load(
            os.path.join('model',f"scores_{model_name}_{model_version}.pth"),map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @staticmethod
    def split_image(img):
        TARGET_SIZE = 512
        Y = list(range(0, img.shape[0]-TARGET_SIZE, TARGET_SIZE))
        Y.append(img.shape[0]-TARGET_SIZE)
        X = list(range(0, img.shape[1]-TARGET_SIZE, TARGET_SIZE))
        X.append(img.shape[1]-TARGET_SIZE)
        img_tile = []
        meta_info = []
        for y in Y:
            for x in X:
                img_tile.append(img[y:y + TARGET_SIZE, x:x + TARGET_SIZE])
                if x == X[-1] and y == Y[-1]:
                    meta_info.append((Y[-2]+TARGET_SIZE,img.shape[0],X[-2]+TARGET_SIZE,img.shape[1]))
                elif x==X[-1]:
                    meta_info.append((y,y+512,X[-2]+TARGET_SIZE,img.shape[1]))
                elif y==Y[-1]:
                    meta_info.append((Y[-2]+TARGET_SIZE,img.shape[0],x,x+512))
                else:
                    meta_info.append((y,y+512,x,x+512))
        
        img_tile = torch.tensor(np.stack(img_tile,axis=0),dtype=torch.float32)/255
        img_tile = img_tile.permute(0,3,1,2)
        return img_tile, meta_info
    

    @staticmethod
    def join_image(mask,meta_info):
        pred_mask = np.zeros((meta_info[-1][1],meta_info[-1][3]))

        for i, info in enumerate(meta_info):
            pred_mask[info[0]:info[1],info[2]:info[3]] = mask[i,:info[1]-info[0],:info[3]-info[2]]
            

        return(pred_mask)
