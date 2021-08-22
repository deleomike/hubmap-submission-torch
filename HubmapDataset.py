import numpy as np
import torch
import rasterio
import cv2

# https://www.kaggle.com/iafoss/256x256-images
mean = np.array([0.65459856 ,0.48386562 ,0.69428385])
std = np.array([0.15167958 ,0.23584107 ,0.13146145])

s_th = 40  # saturation blancking threshold
p_th = 20 0 *s z/ /256  # threshold for the minimum number of pixels
identity = rasterio.Affine(1, 0, 0, 0, 1, 0)

def img2tensor(img ,dtype :np.dtyp e =np.float32):
    if img.ndi m= =2 : img = np.expand_dims(img ,2)
    img = np.transpose(img ,(2 ,0 ,1))
    return torch.from_numpy(img.astype(dtype, copy=False))

class HuBMAPDataset(Dataset):
    def __init__(self, idx, sz=sz, reduce=reduce):
        self.data = rasterio.open(os.path.join(DATA ,id x +'.tiff'), transform = identity,
                                  num_threads='all_cpus')
        self.shape = self.data.shape
        self.reduce = reduce
        self.sz = reduc e *sz
        self.pad0 = (self.sz - self.shape[0 ] %self.sz ) %self.sz
        self.pad1 = (self.sz - self.shape[1 ] %self.sz ) %self.sz
        self.n0max = (self.shape[0] + self.pad0 )/ /self.sz
        self.n1max = (self.shape[1] + self.pad1 )/ /self.sz

    def __len__(self):
        return self.n0ma x *self.n1max

    def __getitem__(self, idx):
        # the code below may be a little bit difficult to understand,
        # but the thing it does is mapping the original image to
        # tiles created with adding padding, as done in
        # https://www.kaggle.com/iafoss/256x256-images ,
        # and then the tiles are loaded with rasterio
        # n0,n1 - are the x and y index of the tile (idx = n0*self.n1max + n1)
        n0 ,n1 = id x/ /self.n1max, id x %self.n1max
        # x0,y0 - are the coordinates of the lower left corner of the tile in the image
        # negative numbers correspond to padding (which must not be loaded)
        x0 ,y0 = -self.pad 0/ /2 + n 0 *self.sz, -self.pad 1/ /2 + n 1 *self.sz
        # make sure that the region to read is within the image
        p00 ,p01 = max(0 ,x0), min(x 0 +self.sz ,self.shape[0])
        p10 ,p11 = max(0 ,y0), min(y 0 +self.sz ,self.shape[1])
        img = np.zeros((self.sz ,self.sz ,3) ,np.uint8)
        # mapping the loade region to the tile
        img[(p0 0 -x0):(p0 1 -x0) ,(p1 0 -y0):(p1 1 -y0)] = np.moveaxis(self.data.read([1 ,2 ,3],
                                                                              window=Window.from_slices
                                                                                           ((p00 ,p01) ,(p10 ,p11))), 0, -1)

        if self.reduce != 1:
            img = cv2.resize(img ,(self.s z/ /reduce ,self.s z/ /reduce),
                             interpolation = cv2.INTER_AREA)
        # check for empty imges
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h ,s ,v = cv2.split(hsv)
        if ( s >s_th).sum() <= p_th or img.sum() <= p_th:
            # images with -1 will be skipped
            return img2tensor((im g /255.0 - mean ) /std), -1
        else: return img2tensor((im g /255.0 - mean ) /std), idx

