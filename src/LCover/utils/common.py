import os
import glob
import cv2
from box.exceptions import BoxValueError
import yaml
from LCover import logger
import json
import joblib
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any
import base64

import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns

import torch
from torch import nn
from torch.utils.data import Dataset

import numpy as np
from pathlib import Path
import torchvision.transforms.functional as TF
import torch.nn.functional as F




@ensure_annotations
def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """reads yaml file and returns

    Args:
        path_to_yaml (str): path like input

    Raises:
        ValueError: if yaml file is empty
        e: empty file

    Returns:
        ConfigBox: ConfigBox type
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")




@ensure_annotations
def load_json(path: Path) -> ConfigBox:
    """load json files data

    Args:
        path (Path): path to json file

    Returns:
        ConfigBox: data as class attributes instead of dict
    """
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return ConfigBox(content)


@ensure_annotations
def save_bin(data: Any, path: Path):
    """save binary file

    Args:
        data (Any): data to be saved as binary
        path (Path): path to binary file
    """
    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data

@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


def decodeImage(imgstring, fileName):
    imgdata = base64.b64decode(imgstring)
    with open(fileName, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImageIntoBase64(croppedImagePath):
    with open(croppedImagePath, "rb") as f:
        return base64.b64encode(f.read())

#############################################################################################################
#############################################################################################################
#############################################################################################################

# The colors of the 5 land uses. Using the colors of the paper
labels_cmap = matplotlib.colors.ListedColormap(["#000000", "#A9A9A9",
        "#8B8680", "#D3D3D3", "#FFFFFF"])



@ensure_annotations
def split_images(DATA_DIR:Path,OUTPUT_DIR:Path,TARGET_SIZE:int):
    """
    A function to split the aerial images into squared images of
    size equal to TARGET_SIZE. Stores the new images into
    a directory named output, located in working directory.
    """
    
    img_paths = glob.glob(os.path.join(DATA_DIR,'images', "*.tif"))
    mask_paths = glob.glob(os.path.join(DATA_DIR,'masks', "*.tif"))

    img_paths.sort()
    mask_paths.sort()
    
    
    for i, (img_path, mask_path) in enumerate(zip(img_paths, mask_paths)):
        img_filename = os.path.splitext(os.path.basename(img_path))[0]
        mask_filename = os.path.splitext(os.path.basename(mask_path))[0]
        img = cv2.imread(img_path)
        mask = cv2.imread(mask_path)

        assert img_filename == mask_filename and img.shape[:2] == mask.shape[:2]

        k = 0
        for y in range(0, img.shape[0], TARGET_SIZE):
            for x in range(0, img.shape[1], TARGET_SIZE):
                img_tile = img[y:y + TARGET_SIZE, x:x + TARGET_SIZE]
                mask_tile = mask[y:y + TARGET_SIZE, x:x + TARGET_SIZE]

                if img_tile.shape[0] == TARGET_SIZE and img_tile.shape[1] == TARGET_SIZE:
                    out_img_path = os.path.join(OUTPUT_DIR, "{}_{}.jpg".format(img_filename, k))
                    cv2.imwrite(out_img_path, img_tile)

                    out_mask_path = os.path.join(OUTPUT_DIR, "{}_{}_m.png".format(mask_filename, k))
                    cv2.imwrite(out_mask_path, mask_tile)

                k += 1

        print("Processed {} {}/{}".format(img_filename, i + 1, len(img_paths)))
    


class SegmentationDataset(Dataset):
    """
    The main class that handles the dataset. Reads the images from
    OUTPUT_DIR, handles the data augmentation transformations and converts
    the numpy images to tensors.
    """
    def __init__(self, DATA_DIR:Path, mode = "train", ratio = None, transforms = None, seed = 42):
        self.mode = mode
        self.transforms = transforms
        self.data_dir = DATA_DIR
        
        if mode in ["train", "test", "val"]:
            with open(os.path.join(self.data_dir, self.mode + ".txt")) as f:
                self.img_names = f.read().splitlines()
                if ratio is not None:
                    print(f"Using the {100*ratio:.2f}% of the initial {mode} set --> {int(ratio*len(self.img_names))}|{len(self.img_names)}")
                    np.random.seed(seed)
                    self.indices = np.random.randint(low = 0, high = len(self.img_names),
                                             size = int(ratio*len(self.img_names)))
                else:
                    print(f"Using the whole {mode} set --> {len(self.img_names)}")
                    self.indices = list(range(len(self.img_names)))
        else:
            raise ValueError(f"mode should be either train, val or test ... not {self.mode}.")
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        if self.transforms is None:
            img = np.transpose(cv2.imread(os.path.join(self.data_dir, self.img_names[self.indices[item]] + ".jpg")),(2,0,1))
            mask = cv2.imread(os.path.join(self.data_dir, self.img_names[self.indices[item]] + "_m.png"))
            label = mask[:,:,1]
        else:
            img = cv2.imread(os.path.join(self.data_dir, self.img_names[self.indices[item]] + ".jpg"))
            mask = cv2.imread(os.path.join(self.data_dir, self.img_names[self.indices[item]] + "_m.png"))
            label = mask[:,:,1]
            transformed = self.transforms(image = img, mask = label)
            img = np.transpose(transformed["image"], (2,0,1))
            label = transformed["mask"]
        del mask
        return torch.tensor(img, dtype = torch.float32)/255, torch.tensor(label, dtype = torch.int64)




# Custom implementation of UNet

##########################################################
##########################################################


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation = 1):
        super(DoubleConv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation,stride=1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding = dilation, stride = 1, bias=False,
                     dilation = dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)
    
class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=5, features = [64, 128, 256, 512],
                rates = (1,1,1,1)):
        super(UNet, self).__init__()
        self.down_part = nn.ModuleList()
        self.up_part = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder Part
        for i,feature in enumerate(features):
            self.down_part.append(DoubleConv(in_channels, feature, dilation = rates[i]))
            in_channels = feature
        # Decoder Part
        for i,feature in enumerate(reversed(features)):
            self.up_part.append(
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)
            )
            self.up_part.append(DoubleConv(2*feature, feature, dilation = rates[i]))
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.output = nn.Conv2d(features[0], out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        skip_connections = []
        for down in self.down_part:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.up_part), 2):
            x = self.up_part[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size = skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection,x), dim = 1)
            x = self.up_part[idx + 1](concat_skip)

        return self.output(x)



#################### Early Stopping ######################

# Source: https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.val_loss_min = val_loss

        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.val_loss_min = val_loss
            self.counter = 0
        

# Custom Implementation of FocalLoss

class FocalLoss(nn.Module):
    def __init__(self, weight = None, gamma = 2, reduction = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.weight = weight

    def forward(self, logits, targets):
        log_prob = F.log_softmax(logits, dim = 1)
        prob = torch.exp(log_prob)
        return F.nll_loss(
            ((1-prob)**self.gamma)*log_prob,
            targets,
            weight = self.weight,
            reduction = self.reduction
        )
    
    
def class_report(classes, scores, acc, jaccard, class_probs):
    print(f"{10*' '}precision{10*' '}recall{10*' '}f1-score{10*' '}support\n")
    acc = float(acc.cpu())
    jaccard = float(jaccard.cpu())
    F1 = []
    for i,target in enumerate(classes):
        precision = float((scores[i,0]/(scores[i,0]+scores[i,1])).cpu())
        recall = float((scores[i,0]/(scores[i,0]+scores[i,3])).cpu())
        f1 = (2*precision*recall)/(precision+recall)
        F1.append(f1)
        print(f"{target}{10*' '}{precision:.2f}{10*' '}{recall:.2f}{10*' '}{f1:.2f}{10*' '}{scores[i,4]}")
    print(f"\n- Total accuracy:{acc:.4f}\n")
    print(f"- Mean IoU: {jaccard:.4f}\n")
    print("- Class probs")
    for idx in class_probs.keys():
        print(f"{classes[idx]}:{class_probs[idx].cpu():.3f}")
    
    return(acc,jaccard,np.array(F1).sum()/len(classes))


def visualize_preds(model, train_set, title, num_samples = 4, seed = 42,
                    w = 10, h = 10, save_title = None, indices = None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    np.random.seed(seed)
    if indices == None:
        indices = np.random.randint(low = 0, high = len(train_set),
                                    size = num_samples)
    sns.set_style("white")
    fig, ax = plt.subplots(figsize = (w,h),
                           nrows = num_samples, ncols = 3)
    model.eval()
    for i,idx in enumerate(indices):
        X,y = train_set[idx]
        X_dash = X[None,:,:,:].to(device)
        preds = torch.argmax(model(X_dash), dim = 1)
        preds = torch.squeeze(preds).detach().cpu().numpy()

        ax[i,0].imshow(np.transpose(X.cpu(), (2,1,0)))
        ax[i,0].set_title("True Image")
        ax[i,0].axis("off")
        ax[i,1].imshow(y, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,1].set_title("Labels")
        ax[i,1].axis("off")
        ax[i,2].imshow(preds, cmap = labels_cmap, interpolation = None,
                      vmin = -0.5, vmax = 4.5)
        ax[i,2].set_title("Predictions")
        ax[i,2].axis("off")
    fig.suptitle(title, fontsize = 20)
    plt.tight_layout()
    if save_title is not None:
        plt.savefig(save_title + ".png")
    plt.show()



