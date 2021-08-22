import os

from fastai.vision.all import *
from models.deeplab.deeplab_v3_plus import DeepLab

dataset = "coco2017"

datasets_path = "../datasets"

coco_path = os.path.join(datasets_path, dataset)

images_path = os.path.join(coco_path, "images")
anno_path = os.path.join(coco_path, "annotations")