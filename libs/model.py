import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as D
import torch.nn.functional as F

import torchvision
from torchvision import transforms as T

import sys
sys.path.append('./')
from libs.arcface import ArcMarginProduct

from tqdm import tqdm


class DenseNet(nn.Module):
    def __init__(self, num_classes=1000, num_channels=6):
        super().__init__()
        preloaded = torchvision.models.densenet169(pretrained=True)
        self.features = preloaded.features
        self.features.conv0 = nn.Conv2d(num_channels, 64, 7, 2, 3)
        #self.classifier = nn.Linear(1920, 512, bias=True)
        self.metric_fc = ArcMarginProduct(1664, num_classes, s=30, m=0.5)
        del preloaded

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        #out= self.classifier(out)
        #out = self.metric_fc(out, target)
        return out

    def arc(self, feature, target):
        return self.metric_fc(feature, target)
