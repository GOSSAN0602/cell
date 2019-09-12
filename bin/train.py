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

import argparse

sys.path.append('./')
from libs.model import DenseNet
from libs.data import ImagesDS
from libs.trainer import trainer
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Training DenseNet201 with Cell Images')
parser.add_argument('--tag',type=str)
parser.add_argument('--lr',type=float)
parser.add_argument('--n_epochs',type=int)
args = parser.parse_args()

#config
path_data='/home/shuki_goto/input/'
device='cuda'
batch_size=16
SAVE_PATH = '/home/shuki_goto/log/'+args.tag+'/'

#define dataset
ds = ImagesDS(path_data+'train.csv', path_data+'imgs')
ds_test = ImagesDS(path_data+'test.csv', path_data+'imgs', mode='test')

#define model
num_classes = 1108
model = DenseNet(num_classes=num_classes)
model.to(device)

#define dataloader
train_loader = D.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = D.DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=2)

# define trainer
trainer=trainer(model, SAVE_PATH=SAVE_PATH, num_epochs=args.n_epochs, lr=args.lr, loader=train_loader)

# train model
trained_model, loss = trainer.train_model()

# save model
torch.save(trained_model.state_dict(), SAVE_PATH+'models/')

# make loss figure
plot_loss(loss, args.n_epochs, SAVE_PATH)
