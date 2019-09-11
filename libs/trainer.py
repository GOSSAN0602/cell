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

class trainer(model):
    def __init__(self, SAVE_PATH, model, num_epochs, lr, loader):
        self.SAVE_PATH = SAVE_PATH
        self.criterion = nn.BCEWithLogitsLoss()
        self.model=model
        self.num_epochs = num_epochs
        self.loader=loader
        optimizer = torch.optim.Adam(model.parameters(lr=lr))

    def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return np.array(res)

    def train_model():
        loss=[]
        for epoch in range(self.num_epochs):
            tloss = 0
            acc = np.zeros(1)
            for x, y in self.loader:
                x = x.to(device)
                optimizer.zero_grad()
                output = self.model(x)
                target = torch.zeros_like(output, device=device)
                target[np.arange(x.size(0)), y] = 1
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                tloss += loss.item()
                acc += accuracy(output.cpu(), y)
                del loss, output, y, x, target
            # Save Model
            torch.save(model.state_dict(), self.SAVE_PATH+str(epoch)+'.pth')
            loss.append(tloss.to_cpu())
            print('Epoch {} -> Train Loss: {:.4f}, ACC: {:.2f}%'.format(epoch+1, tloss/tlen, acc[0]/tlen))
        return np.array(loss)
