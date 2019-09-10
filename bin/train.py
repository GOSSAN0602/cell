import numpy as np
import pandas as pd
from PIL import Image
import sys

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T

sys.path.append('./')
from libs.model import DenseNet

from tqdm import tqdm

#config
path_data='../input/'
device='cuda'
batch_size=16

#define dataset
ds = ImageDS(path_data+'/train.csv', path_data)
ds_test = ImageDS(path_data+'/test.csv', path_data, mode='test')

#define model
num_classes = 1108
model = DenseNet(num_classes=num_classes)
model.to(device)

#define dataloader
train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

# train model
trainer=trainer(model,num_epochs=args.num_epochs,lr=args.lr,loader=train_loader)
trained_model, loss = trainer.train_model()

# save model & Loss
trained_model.save()
loss.to_npy('')
