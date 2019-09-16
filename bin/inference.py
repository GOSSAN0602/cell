import numpy as np
import pandas as pd
from PIL import Image
import sys
import os

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

import argparse

sys.path.append('./')
from libs.model import DenseNet
from libs.data import ImagesDS
from libs.trainer import trainer
from libs.plot import plot_loss
from tqdm import tqdm


#define dataset
ds_test = ImagesDS(path_data+'test.csv', path_data+'imgs', mode='test')

#define model
num_classes = 1108
model = DenseNet(num_classes=num_classes)
model.load_weight('../log/densenet201_baseline')
model.to(device)
model = torch.nn.DataParallel(model) # make parallel

#define dataloader
test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)


