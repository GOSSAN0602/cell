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

from tqdm import tqdm
import argparse

sys.path.append('./')
from libs.model import DenseNet
from libs.data import ImagesDS
from libs.trainer import trainer
from libs.plot import plot_loss

def calcurate_center_train_feature(model, train_loader):
    device='cuda'
    #i=0
    center_feature=np.zeros([1108, 1664])
    for x, y in tqdm(train_loader):
     #   i+=1
      #  if i == 2:
       #     break
        feature = np.array(model(x.to(device)).cpu().detach())
        center_feature[y]+=feature
    train_tb=pd.read_csv('/home/shuki_goto/input/train.csv')
    num_class = np.array(train_tb['sirna'].value_counts()).reshape(1108,1)
    center_feature /= num_class
    return center_feature

def pred(tcf, model, test_loader):
    device='cuda'
    preds = np.empty(0)
    for x, _ in test_loader:
        x = x.to(device)
        test_feature = np.array(model(x).cpu().detach())
        train_test_similarity = cosine_similarity(test_feature, tcf)
        idx = train_test_similarity.argmax(axis=1)
        preds = np.append(preds, idx, axis=0)
    return preds
