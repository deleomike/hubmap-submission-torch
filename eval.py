import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import tifffile as tiff
import cv2
import os
import gc
from tqdm import tqdm
import rasterio
from rasterio.windows import Window

from fastai.vision.all import *
from torch.utils.data import Dataset, DataLoader
from HubmapModel import DeepLabv3_plus as HubmapModel

import warnings

warnings.filterwarnings("ignore")

sz = 512  # the size of tiles
reduce = 4  # reduce the original images by 4 times
TH = 0.41  # threshold for positive predictions
DATA = './hubmap-kidney-segmentation/test/'
if sz == 256:
    MODELS = [f'model_{i}.pth' for i in range(4)]
    bs = 64
elif sz == 512:
    MODELS = [f'model_2_{i}.pth' for i in range(4)]
    bs = 32
elif sz == 1024:
    MODELS = [f'model_3_{i}.pth' for i in range(4)]
    bs = 24
print("Batch size", bs)

df_sample = pd.read_csv('./hubmap-kidney-segmentation/sample_submission.csv')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# functions to convert encoding to mask and mask to encoding
def enc2mask(encs, shape):
    img = np.zeros(shape[0] * shape[1], dtype=np.uint8)
    for m, enc in enumerate(encs):
        if isinstance(enc, np.float) and np.isnan(enc): continue
        s = enc.split()
        for i in range(len(s) // 2):
            start = int(s[2 * i]) - 1
            length = int(s[2 * i + 1])
            img[start:start + length] = 1 + m
    return img.reshape(shape).T


def mask2enc(mask, n=1):
    pixels = mask.T.flatten()
    encs = []
    for i in range(1, n + 1):
        p = (pixels == i).astype(np.int8)
        if p.sum() == 0:
            encs.append(np.nan)
        else:
            p = np.concatenate([[0], p, [0]])
            runs = np.where(p[1:] != p[:-1])[0] + 1
            runs[1::2] -= runs[::2]
            encs.append(' '.join(str(x) for x in runs))
    return encs


# https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
# with transposed mask
def rle_encode_less_memory(img):
    # the image should be transposed
    pixels = img.T.flatten()

    # This simplified method requires first and last pixel to be zero
    pixels[0] = 0
    pixels[-1] = 0
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 2
    runs[1::2] -= runs[::2]

    return ' '.join(str(x) for x in runs)


# https://www.kaggle.com/iafoss/256x256-images
if sz == 256:
    mean = np.array([0.65459856, 0.48386562, 0.69428385])
    std = np.array([0.15167958, 0.23584107, 0.13146145])

# 512x512
elif sz == 512:
    mean = np.array([0.64246083, 0.48809628, 0.68370845])
    std = np.array([0.17817486, 0.24006252, 0.16265069])

# 1024x1024
else:
    mean = np.array([0.63244577, 0.49794695, 0.66810544])
    std = np.array([0.22345657, 0.26747085, 0.21520419])


s_th = 40  # saturation blancking threshold
p_th = 1000 * (sz // 256) ** 2  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2: img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(DATA, idx + '.tiff'), transform=identity,
                                  num_threads='all_cpus')
        # some images have issues with their format
        # and must be saved correctly before reading with rasterio
        if self.data.count != 3:
            subdatasets = self.data.subdatasets
            self.layers = []
            if len(subdatasets) > 0:
                for i, subdataset in enumerate(subdatasets, 0):
                    self.layers.append(rasterio.open(subdataset))
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduce * sz
        self.pad0 = (self.sz - self.shape[0] % self.sz) % self.sz
        self.pad1 = (self.sz - self.shape[1] % self.sz) % self.sz
        self.n0max = (self.shape[0] + self.pad0) // self.sz
        self.n1max = (self.shape[1] + self.pad1) // self.sz

    def __len__(self):
        return self.n0max * self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0, n1 = idx // self.n1max, idx % self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0, y0 = -self.pad0 // 2 + n0 * self.sz, -self.pad1 // 2 + n1 * self.sz
        # make sure that the region to read is within the image
        p00, p01 = max(0, x0), min(x0 + self.sz, self.shape[0])
        p10, p11 = max(0, y0), min(y0 + self.sz, self.shape[1])
        img = np.zeros((self.sz, self.sz, 3), np.uint8)
        # mapping the loade region to the tile
        if self.data.count == 3:
            img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0)] = np.moveaxis(self.data.read([1, 2, 3],
                                                                                           window=Window.from_slices(
                                                                                               (p00, p01), (p10, p11))),
                                                                            0, -1)
        else:
            for i, layer in enumerate(self.layers):
                img[(p00 - x0):(p01 - x0), (p10 - y0):(p11 - y0), i] = \
                    layer.read(1, window=Window.from_slices((p00, p01), (p10, p11)))

        if self.reduce != 1:
            img = cv2.resize(img, (self.sz // reduce, self.sz // reduce),
                             interpolation=cv2.INTER_AREA)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if (s > s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((img / 255.0 - mean) / std), -1
        else:
            return img2tensor((img / 255.0 - mean) / std), idx


# iterator like wrapper that returns predicted masks
class Model_pred:
    def __init__(self, models, dl, tta: bool = True, half: bool = False):
        self.models = models
        self.dl = dl
        self.tta = tta
        self.half = half

    def __iter__(self):
        count = 0
        with torch.no_grad():
            for x, y in iter(self.dl):
                if ((y >= 0).sum() > 0):  # exclude empty images
                    x = x[y >= 0].to(device)
                    y = y[y >= 0]
                    if self.half: x = x.half()
                    py = None
                    for model in self.models:
                        p = model(x)
                        p = torch.sigmoid(p).detach()
                        if py is None:
                            py = p
                        else:
                            py += p
                    if self.tta:
                        # x,y,xy flips as TTA
                        flips = [[-1], [-2], [-2, -1]]
                        for f in flips:
                            xf = torch.flip(x, f)
                            for model in self.models:
                                p = model(xf)
                                p = torch.flip(p, f)
                                py += torch.sigmoid(p).detach()
                        py /= (1 + len(flips))
                    py /= len(self.models)

                    py = F.upsample(py, scale_factor=reduce, mode="bilinear")
                    py = py.permute(0, 2, 3, 1).float().cpu()

                    batch_size = len(py)
                    for i in range(batch_size):
                        yield py[i], y[i]
                        count += 1

    def __len__(self):
        return len(self.dl.dataset)


models = []
for path in MODELS:
    print(path)
    state_dict = torch.load(path, map_location=torch.device(device))
    model = HubmapModel(os=16)
    model.load_state_dict(state_dict)
    model.float()
    model.eval()
    model.to(device)
    models.append(model)

del state_dict

names, preds = [], []

names, preds = [], []
for idx, row in tqdm(df_sample.iterrows(), total=len(df_sample)):
    idx = row['id']
    ds = HuBMAPDataset(idx)
    # rasterio cannot be used with multiple workers
    dl = DataLoader(ds, bs, num_workers=0, shuffle=False, pin_memory=True)
    mp = Model_pred(models, dl)
    # generate masks
    mask = torch.zeros(len(ds), ds.sz, ds.sz, dtype=torch.int8)
    for p, i in iter(mp): mask[i.item()] = p.squeeze(-1) > TH

    # reshape tiled masks into a single mask and crop padding
    mask = mask.view(ds.n0max, ds.n1max, ds.sz, ds.sz). \
        permute(0, 2, 1, 3).reshape(ds.n0max * ds.sz, ds.n1max * ds.sz)
    mask = mask[ds.pad0 // 2:-(ds.pad0 - ds.pad0 // 2) if ds.pad0 > 0 else ds.n0max * ds.sz,
           ds.pad1 // 2:-(ds.pad1 - ds.pad1 // 2) if ds.pad1 > 0 else ds.n1max * ds.sz]

    # convert to rle
    # https://www.kaggle.com/bguberfain/memory-aware-rle-encoding
    rle = rle_encode_less_memory(mask.numpy())
    names.append(idx)
    preds.append(rle)
    del mask, ds, dl

df = pd.DataFrame({'id': names, 'predicted': preds})
df.to_csv('submission.csv', index=False)
