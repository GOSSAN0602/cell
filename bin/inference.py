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

from sklearn.metrics.pairwise import cosine_similarity

import argparse

sys.path.append('./')
from libs.model import DenseNet
from libs.data import ImagesDS
from libs.trainer import trainer
from libs.plot import plot_loss
from libs.pred_utils import calcurate_center_train_feature, pred
from tqdm import tqdm

# config
path_data = '/home/shuki_goto/input/'
device='cuda'
batch_size=33

#define dataset
ds = ImagesDS(path_data+'train.csv', path_data+'imgs')
ds_test = ImagesDS(path_data+'test.csv', path_data+'imgs', mode='test')

#define model
num_classes = 1108
model = DenseNet(num_classes=num_classes)
model.load_state_dict(torch.load('/home/shuki_goto/log/dense169_arcface_baseline/16.pth'), strict=False)
model.eval()
model.to(device)
model = torch.nn.DataParallel(model) # make parallel

#define dataloader
train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

train_center_feature = calcurate_center_train_feature(model,train_loader)

preds = pred(train_center_feature, model, test_loader)

submission = pd.read_csv(path_data + '/test.csv')
submission['sirna'] = preds.astype(int)
submission.to_csv('submission.csv', index=False, columns=['id_code','sirna'])
