from __future__ import print_function, division

import torchvision
import random
import torch
import os
import cv2
import torch

from fastai.vision.all import *
import fastai
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from lovasz import lovasz_hinge
from albumentations import *
from torch.autograd import Variable
from HubmapModel import DeepLabv3_plus as HubmapModel
from HubmapModel import split_layers as hm_splitter
from models.deeplab.deeplab_v3_plus import DeepLab
from tqdm import tqdm
try:
    from itertools import  ifilterfalse
except ImportError: # py3k
    from itertools import  filterfalse
from focal_loss import FocalLoss

import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

#512 th is 0.433

sz = 256
# 48 for 512
if sz == 256:
    bs = 190
elif sz == 512:
    bs = 48 #batch size
elif sz == 1024:
    bs = 12 #batch size
else:
    bs = 12
# 12 for 1024
# Input Parameters
nfolds = 4
fold = 0
SEED = 2020
TRAIN = "./input/hubmap-{}x{}/train/".format(sz, sz)
MASKS = "./input/hubmap-{}x{}/masks/".format(sz, sz)
TRAIN = "./data/{}x{}/train/".format(sz, sz)
MASKS = "./data/{}x{}/masks/".format(sz, sz)
LABELS = './hubmap-kidney-segmentation/train.csv'
NUM_WORKERS = 16
device = "cuda"


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


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, fold=fold, train=True, tfms=None, cache=True):
        ids = pd.read_csv(LABELS).id.values
        kf = KFold(n_splits=nfolds, random_state=SEED, shuffle=True)
        ids = set(ids[list(kf.split(ids))[fold][0 if train else 1]])
        self.fnames = [fname for fname in os.listdir(TRAIN) if fname.split('_')[0] in ids]
        self.train = train
        self.tfms = tfms
        self.cache = cache

        if self.cache:
            self.images = [cv2.cvtColor(cv2.imread(os.path.join(TRAIN, fname)), cv2.COLOR_BGR2RGB)
                             for fname in tqdm(self.fnames)]
            self.masks = [cv2.imread(os.path.join(MASKS, fname), cv2.IMREAD_GRAYSCALE)
                              for fname in tqdm(self.fnames)]

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, idx):
        fname = self.fnames[idx]

        if self.cache:
            img = self.images[idx]
            mask = self.masks[idx]
        else:
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
        res = dices.max()
        if res.item() < 0.001:
            print("uh oh")
        return res


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

fl = FocalLoss()
dice = Dice_th_pred(np.arange(0.2, 0.7, 0.01))
print("Training Phase")
for fold in range(nfolds):
    ds_t = HuBMAPDataset(fold=fold, train=True, tfms=get_aug())
    ds_v = HuBMAPDataset(fold=fold, train=False)
    print("Datasets Created")
    data = ImageDataLoaders.from_dsets(ds_t, ds_v, bs=bs,
                                       num_workers=NUM_WORKERS, pin_memory=True).cuda()
    print("Data Loader created")
    # model = HubmapModel(n_classes=1, pretrained=False, os=8)
    model = DeepLab(num_classes=1, backbone="drn", output_stride=8)

    split_layers = lambda m: [list(model.get_1x_lr_params()), list(model.get_10x_lr_params())]
    print("Model Loaded")
    state_dict = torch.load("./models/deeplab/pretrained/deeplab-drn.pth")["state_dict"]
    model.load_my_state_dict(state_dict)

    # state_dict = torch.load("./model_drn_512_{}.pth".format(fold))
    # model.load_state_dict(state_dict)
    learn = Learner(dls=data, model=model, loss_func=symmetric_lovasz,
                    metrics=[Dice_soft(), Dice_th()],
                    splitter=split_layers).to_fp16()
    print("Learner Created")
    # start with training the head
    learn.freeze_to(-1)  # doesn't work
    for param in learn.opt.param_groups[0]['params']:
        param.requires_grad = False
    # tmp = learn.lr_find()
    # print(tmp)
    learn.fit_one_cycle(4, lr_max=0.5e-2)
    # learn.fit_one_cycle(16, lr_max=0.5e-2)

    # continue training full model
    learn.unfreeze()
    learn.fit_one_cycle(64, lr_max=slice(2e-3, 2e-2),
                        cbs=SaveModelCallback(monitor='dice_th', comp=np.greater))
    torch.save(learn.model.state_dict(), f'model_drn_{sz}_{fold}.pth')

    try:
        del ds_t
        del ds_v
    except:
        pass
    # model evaluation on val and saving the masks
    mp = Model_pred(learn.model, learn.dls.loaders[1])
    with zipfile.ZipFile('val_masks_tta.zip', 'a') as out:
        for p in progress_bar(mp):
            dice.accumulate(p[0], p[1])
            save_img(p[0], p[2], out)



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



#TODO: Acknowledgement
"""
Investigators using HuBMAP data in publications or presentations are requested to cite The Human Body at Cellular Resolution: the NIH Human BioMolecular Atlas Program (doi:10.1038/s41586-019-1629-x) and to include an acknowledgement of HuBMAP. Suggested language for such an acknowledgment is: “The results here are in whole or part based upon data generated by the NIH Human BioMolecular Atlas Program (HuBMAP): https://hubmapconsortium.org.”
"""

# names, preds = [], []
# for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
#     idx = row['id']
#     ds = HuBMAPDataset(idx)
#     # rasterio cannot be used with multiple workers
#     dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
#     mp = Model_pred(models, dl)
#     # generate masks
#     mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
#     for p, i in iter(mp): mask[i.item()] = p.squeeze(-1) > TH
#
#     # reshape tiled masks into a single mask and crop padding
#     mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz). \
#         permute(0, 2, 1, 3).reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
#     mask = mask[ds.pad0 // 2:-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
#            ds.pad1 // 2:-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz]
#
#     # convert to rle
#     # https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
#     rle = rle_encode_less_memory(mask.numpy())
#     names.append(idx)
#     preds.append(rle)
#     del mask, ds, dl

