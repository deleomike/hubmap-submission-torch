#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


from torchvision import transforms as py_transforms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
import torch
import random
import os
import torchmetrics

import matplotlib.pyplot as plt

import pandas as pd
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset, DataLoader
from lovasz import lovasz_hinge
from albumentations import *
from sklearn.model_selection import KFold
from tqdm.notebook import tqdm
from pytorch_lightning.callbacks.early_stopping import EarlyStopping


# In[2]:



sz = 256

def getBatchSize(sz):

    # 48 for 512
    if sz == 256:
        bs = 256
    elif sz == 512:
        bs = 64 #batch size
    elif sz == 1024:
        bs = 16 #batch size
    else:
        bs = 12
        
    return bs

def getPaths(sz):
    TRAIN = "./data/{}x{}/train/".format(sz, sz)
    MASKS = "./data/{}x{}/masks/".format(sz, sz)
    # TRAIN = "./kaggle/data/{}x{}/train/".format(sz, sz)
    # MASKS = "./data/{}x{}/masks/".format(sz, sz)
    LABELS = "./hubmap-kidney-segmentation/train.csv"
    
    return TRAIN, MASKS, LABELS

# 12 for 1024
# Input Parameters
nfolds = 4
fold = 0
SEED = 2020
NUM_WORKERS = 16


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # torch.backends.
    torch.backends.cudnn.deterministic = True
    # the following line gives ~10% speedup
    # but may lead to some stochasticity in the results
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

def getStats(sz):

    # https://www.kaggle.com/iafoss/256x256-images
    if sz == 256:
        mean = np.array([0.62950801, 0.45658695, 0.67636472])
        std = np.array([0.15937661, 0.21920461, 0.14160065])

    # 512x512
    elif sz == 512:
        mean = np.array([0.64246083, 0.48809628, 0.68370845])
        std = np.array([0.17817486, 0.24006252, 0.16265069])

    # 1024x1024
    else:
        mean = np.array([0.63244577, 0.49794695, 0.66810544])
        std = np.array([0.22345657, 0.26747085, 0.21520419])
        
    return mean, std


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


# In[ ]:





# In[3]:



class HuBMAPDataset(Dataset):
    def __init__(self, mean, std, train_path, mask_path, label_path, fold=fold, train=True, tfms=None, cache=True):
        self.TRAIN = train_path
        self.LABELS = label_path
        self.MASKS = mask_path
        
        
        ids = pd.read_csv(self.LABELS).id.values
        kf = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(self.TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms
        self.cache = cache
        
        self.mean = mean
        self.std = std

        if self.cache:
            self.images = [cv2.cvtColor(cv2.imread(os.path.join(self.TRAIN, fname)), cv2.COLOR_BGR2RGB)
                             for fname in tqdm(self.fnames)]
            self.masks = [cv2.imread(os.path.join(self.MASKS, fname), cv2.IMREAD_GRAYSCALE)
                              for fname in tqdm(self.fnames)]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        if self.cache:
            img = self.images[idx]
            mask = self.masks[idx]
        else:
            img = cv2.cvtColor(cv2.imread(os.path.join(self.TRAIN, fname)), cv2.COLOR_BGR2RGB)
            mask = cv2.imread(os.path.join(self.MASKS, fname), cv2.IMREAD_GRAYSCALE)

        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img / 255 - self.mean) / self.std ), img2tensor(mask)


# In[4]:


def get_aug(p=1.0):

    # return Compose([
    #     HorizontalFlip(),
    #     VerticalFlip()
    # ])

    return Compose([
        HorizontalFlip(),
        VerticalFlip(),
        RandomRotate90(),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=15, p=0.9,
                         border_mode=cv2.BORDER_REFLECT),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.3),
        OneOf([
            HueSaturationValue(10, 15, 10),
            CLAHE(clip_limit=2),
            RandomBrightnessContrast(),
        ], p=0.3),
    ], p=p)


# In[5]:


def createDataset(size, folds, fold, train, tfms=None):
    mean, std = getStats(size)
    
    train_path, mask_path, label_path = getPaths(size)
    
    ds = HuBMAPDataset(mean=mean, std=std, train_path=train_path, mask_path=mask_path, label_path=label_path,
                       fold=fold, tfms=tfms, cache=False, train=train)
    
    return ds


def createDataLoaders(size, folds, fold):
    
    ds_train = createDataset(size=size, folds=folds, fold=fold, train=True, tfms=None)
    ds_val = createDataset(size=size, folds=folds, fold=fold, train=False)
    
    bs = getBatchSize(size)
    
    train_loader = DataLoader(ds_train, batch_size=bs, num_workers=0, shuffle=True)
    val_loader = DataLoader(ds_val, batch_size=bs, num_workers=0)
    
    return train_loader, val_loader

createDataset(256, 4, False, 0)

createDataLoaders(256, 4, 0)


# In[11]:


class HubmapSystem(pl.LightningModule):
    
    def __init__(self, learning_rate=1.3182567385564074e-07):
        super().__init__()
        
        self.learning_rate = learning_rate
        
#         print("Learning Rate: ", self.learning_rate)
        
        self.num_classes = 1
        
        self.epoch_dice = []

        self.val_loss = []
        self.val_miou = []
        self.val_mdice = []
        
        
#         aux_params=dict(
#             pooling='avg',             # one of 'avg', 'max'
#             dropout=0.5,               # dropout ratio, default is None
#             activation='sigmoid',      # activation function, default is None
#             classes=self.num_classes,                 # define number of output labels
#         )
        
        self.model = smp.FPN(
            encoder_name="efficientnet-b3",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
            in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=self.num_classes,                      # model output channels (number of classes in your dataset)
            decoder_dropout=0.1,
        )

        self.iou = torchmetrics.IoU(num_classes=self.num_classes + 1)

        # self.model.to(dtype=torch.float16)
        
        # if self.precision == 16:
        #
        #     self.model.to(dtype=torch.float16)
        # else:
        #     self.model.to(dtype=torch.float32)
        
#         for name, param in self.model.named_parameters():
#             param.requires_grad = False
            
#         for name, param in self.model.segmentation_head.named_parameters():
#             param.requires_grad = True
        
#         for name, parameter in self.model.named_parameters():
#             print(name, parameter.requires_grad)
    
    def forward(self, x):
        res = self.model(x)
        return res
    
    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        y_hat = self(x)
#         print(y_hat)
        loss = self.symmetric_lovasz(y_hat, y)
        with torch.no_grad():
            self.log("train_loss", loss)
            self.log("dice", self.dice_metric(y_hat, y), prog_bar=True)
            self.log("iou", self.iou_metric(y_hat, y), prog_bar=True)
        
        self.epoch_dice.append(self.dice_metric(y_hat, y).item())
        
#         print(torchmetrics.functional.dice_score(y_hat, y))
        return loss

    def on_train_epoch_end(self, outputs):
        avg = np.mean(self.epoch_dice)
        self.log("MEAN DICE", avg, prog_bar=True)
        self.epoch_dice = []

    def on_validation_end(self) -> None:
        # self.log("epoch_val_loss", np.mean(self.val_loss), prog_bar=True)
        # self.log("val_mdice", np.mean(self.val_mdice), prog_bar=True)
        # self.log("val_miou", np.mean(self.val_miou), prog_bar=True)
        self.val_miou = []
        self.val_mdice = []
        self.val_loss = []

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self(x)
        # print("yhat:", y_hat)
        # print("y", y)
        loss = self.symmetric_lovasz(y_hat, y)
        # print(loss)
        self.val_loss.append(loss.cpu())
        self.val_mdice.append(self.dice_metric(y_hat, y).cpu())
        self.val_miou.append(self.iou_metric(y_hat, y).cpu())
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_dice", self.dice_metric(y_hat, y), prog_bar=True)
        self.log("val_iou", self.iou_metric(y_hat, y), prog_bar=True)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
        
    def symmetric_lovasz(self, outputs, targets):
        return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))

    def iou_metric(self, y_hat, y):
        def helper(t):
            t = t.squeeze()
            t = torch.tanh(t)
            blank = torch.zeros_like(t)
            mean = torch.mean(t, dim=(1, 2))
            std = torch.std(t, dim=(1, 2))

            # do the mean
            threshold = mean + std

            # get the size
            size = t.shape[2]
            # get the batch
            bs = t.shape[0]

            stacked = torch.stack([threshold] * (size ** 2)).T

            threshold = stacked.view([bs, size, size])

            mask = threshold < t

            blank[mask] = 1
            blank = blank.to(dtype=torch.int8)
            return blank

        y_hat_m = helper(y_hat)
        y_m = helper(y)

        return self.iou(y_hat_m, y_m)

    
    def dice_metric(self, y_hat, y):
        return torchmetrics.functional.f1(y_hat, y, num_classes=self.num_classes) 


def plot3d(tensor):
    sz = tensor.shape[1]
    x = torch.arange(sz)
    y = torch.arange(sz)

    X, Y = np.meshgrid(x, y)

    fig = plt.figure()
    ax = plt.axes(projection="3d")

    ax.contour3D(X, Y, tensor.cpu(), 100)

# In[12]:


#ds_train = HuBMAPDataset(fold=0, train=True, tfms=get_aug(), cache=False)
#ds_val = HuBMAPDataset(fold=0, train=False, cache=False)


# In[13]:


#train_loader = DataLoader(ds_train, batch_size=bs, num_workers=16)
#val_loader = DataLoader(ds_val, batch_size=bs, num_workers=16)


# In[14]:



#system = HubmapSystem()
#trainer = pl.Trainer(gpus=1, precision=16, deterministic=True,
#                     benchmark=True, check_val_every_n_epoch=1, accumulate_grad_batches=22)
#trainer.fit(system, train_loader, val_loader)


# In[15]:

def train_one_cyle(size, fold, model=None):
    print("========================{}==============================".format(size))
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=4,
        verbose=True,
        mode='min'
    )

    system = HubmapSystem()
    if model is not None:
        system.model = model

    trainer = pl.Trainer(gpus=1, precision=16, deterministic=True, callbacks=[early_stop_callback],
                         benchmark=True, check_val_every_n_epoch=1, stochastic_weight_avg=True)

    train_loader, val_loader = createDataLoaders(size, 4, fold)

    # Run learning rate finder
    lr_finder = trainer.tuner.lr_find(system, train_loader, val_loader)

    system.learning_rate = lr_finder.suggestion()
    print("LR: ", system.learning_rate)

    trainer.fit(system, train_loader, val_loader)

    return system.model

if __name__=="__main__":

    for fold in range(nfolds):
        print("FOLD: ", fold)

        model = train_one_cyle(256, fold)

        model = train_one_cyle(512, fold, model)

        model = train_one_cyle(1024, fold, model)





# In[ ]


# In[ ]:




