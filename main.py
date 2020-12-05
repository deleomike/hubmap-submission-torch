from __future__ import print_function, division

import torchvision
import random
import torch
import os
import cv2
import torch

from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lovasz import lovasz_hinge
from albumentations import *
from torch.autograd import Variable
from HubmapModel import HubmapModel
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse

import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# Input Parameters
bs = 64 #batch size
nfolds = 4
fold = 0
SEED = 2020
TRAIN = './input/hubmap-256x256/train/'
MASKS = './input/hubmap-256x256/masks/'
LABELS = './input/hubmap-kidney-segmentation/train.csv'
NUM_WORKERS = 4


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    # the following line gives ~10% speedup
    # but may lead to some stochasticity in the results
    torch.backends.cudnn.benchmark = True


seed_everything(SEED)

# https://www.kaggle.com/iafoss/256x256-images
mean = np.array([0.65459856, 0.48386562, 0.69428385])
std = np.array([0.15167958, 0.23584107, 0.13146145])


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, fold=fold, train=True, tfms=None):
        ids = pd.read_csv(LABELS).id.values
        kf = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]
        img = cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(MASKS, fname), cv2.IMREAD_GRAYSCALE)
        if self.tfms is not None:
            augmented = self.tfms(image=img, mask=mask)
            img, mask = augmented['image'], augmented['mask']
        return img2tensor((img / 255.0 - mean) / std), img2tensor(mask)


def get_aug(p=1.0):
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




def symmetric_lovasz(outputs, targets):
    return 0.5*(lovasz_hinge(outputs, targets) + lovasz_hinge(-outputs, 1.0 - targets))


class Dice_soft(Metric):
    def __init__(self, axis=1):
        self.axis = axis

    def reset(self): self.inter, self.union = 0, 0

    def accumulate(self, learn):
        pred, targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        self.inter += (pred * targ).float().sum().item()
        self.union += (pred + targ).float().sum().item()

    @property
    def value(self): return 2.0 * self.inter / self.union if self.union > 0 else None


# dice with automatic threshold selection
class Dice_th(Metric):
    def __init__(self, ths=np.arange(0.1, 0.9, 0.05), axis=1):
        self.axis = axis
        self.ths = ths

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, learn):
        pred, targ = flatten_check(torch.sigmoid(learn.pred), learn.y)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p * targ).float().sum().item()
            self.union[i] += (p + targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0,
                            2.0 * self.inter / self.union, torch.zeros_like(self.union))
        return dices.max()


# iterator like wrapper that returns predicted and gt masks
class Model_pred:
    def __init__(self, model, dl, tta: bool = True, half: bool = False):
        self.model = model
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        self.model.eval()
        name_list = self.dl.dataset.fnames
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                x = x.cuda()
                if self.half: x = x.half()
                p = self.model(x)
                py = torch.sigmoid(p).detach()
                if self.tta:
                    # x,y,xy flips as TTA
                    flips = [[-1], [-2], [-2, -1]]
                    for f in flips:
                        p = self.model(torch.flip(x, f))
                        p = torch.flip(p, f)
                        py += torch.sigmoid(p).detach()
                    py /= (1 + len(flips))
                if y is not None and len(y.shape) == 4 and py.shape != y.shape:
                    py = F.upsample(py, size=(y.shape[-2], y.shape[-1]), mode="bilinear")
                py = py.permute(0, 2, 3, 1).float().cpu()
                batch_size = len(py)
                for i in range(batch_size):
                    target = y[i].detach().cpu() if y is not None else None
                    yield py[i], target, name_list[count]
                    count += 1

    def __len__(self):
        return len(self.dl.dataset)


class Dice_th_pred(Metric):
    def __init__(self, ths=np.arange(0.1, 0.9, 0.01), axis=1):
        self.axis = axis
        self.ths = ths
        self.reset()

    def reset(self):
        self.inter = torch.zeros(len(self.ths))
        self.union = torch.zeros(len(self.ths))

    def accumulate(self, p, t):
        pred, targ = flatten_check(p, t)
        for i, th in enumerate(self.ths):
            p = (pred > th).float()
            self.inter[i] += (p * targ).float().sum().item()
            self.union[i] += (p + targ).float().sum().item()

    @property
    def value(self):
        dices = torch.where(self.union > 0.0, 2.0 * self.inter / self.union,
                            torch.zeros_like(self.union))
        return dices


def save_img(data, name, out):
    data = data.float().cpu().numpy()
    img = cv2.imencode('.png', (data * 255).astype(np.uint8))[1]
    out.writestr(name, img)

#split the model to encoder and decoder for fast.ai
split_layers = lambda m: [list(m.enc0.parameters())+list(m.enc1.parameters())+
                list(m.enc2.parameters())+list(m.enc3.parameters())+
                list(m.enc4.parameters()),
                list(m.aspp.parameters())+list(m.dec4.parameters())+
                list(m.dec3.parameters())+list(m.dec2.parameters())+
                list(m.dec1.parameters())+list(m.fpn.parameters())+
                list(m.final_conv.parameters())]

dice = Dice_th_pred(np.arange(0.2, 0.7, 0.01))
for fold in range(nfolds):
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=get_aug())
    ds_v = HuBMAPDataset(fold=fold, train=False)
    data = ImageDataLoaders.from_dsets(ds_t, ds_v, bs=bs,
                                       num_workers=NUM_WORKERS, pin_memory=True).cuda()
    model = HubmapModel()
    learn = Learner(dls=data, model=model, loss_func=symmetric_lovasz,
                    metrics=[Dice_soft(), Dice_th()],
                    splitter=split_layers).to_fp16(clip=0.5)
    # start with training the head
    learn.freeze_to(-1)  # doesn't work
    for param in learn.opt.param_groups[0]['params']:
        param.requires_grad = False
    learn.fit_one_cycle(6, lr_max=0.5e-2)

    # continue training full model
    learn.unfreeze()
    learn.fit_one_cycle(32, lr_max=slice(2e-4, 2e-3),
                        cbs=SaveModelCallback(monitor='dice_th', comp=np.greater))
    torch.save(learn.model.state_dict(), f'model_{fold}.pth')

    # model evaluation on val and saving the masks
    mp = Model_pred(learn.model, learn.dls.loaders[1])
    with zipfile.ZipFile('val_masks_tta.zip', 'a') as out:
        for p in progress_bar(mp):
            dice.accumulate(p[0], p[1])
            save_img(p[0], p[2], out)
    gc.collect()




dices = dice.value
noise_ths = dice.ths
best_dice = dices.max()
best_thr = noise_ths[dices.argmax()]
plt.figure(figsize=(8,4))
plt.plot(noise_ths, dices, color='blue')
plt.vlines(x=best_thr, ymin=dices.min(), ymax=dices.max(), colors='black')
d = dices.max() - dices.min()
plt.text(noise_ths[-1]-0.1, best_dice-0.1*d, f'DICE = {best_dice:.3f}', fontsize=12);
plt.text(noise_ths[-1]-0.1, best_dice-0.2*d, f'TH = {best_thr:.3f}', fontsize=12);
plt.show()



