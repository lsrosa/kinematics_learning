import time, math, sys
import numpy as np
from pathlib import Path as path
from matplotlib import pyplot as plt

from utils import *
import torch as t

class NN(t.nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.nn1 = t.nn.Sequential()
        self.nn1.append(t.nn.Linear(1, 4))
        self.nn1.append(t.nn.ReLU())
        self.nn1.append(t.nn.Linear(4, 1))
        
        self.nn2 = t.nn.Sequential()
        self.nn2.append(t.nn.Linear(1, 4))
        self.nn2.append(t.nn.ReLU())
        self.nn2.append(t.nn.Linear(4, 1))

        self.optim1 = t.optim.SGD(self.nn1.parameters(), 1e-6)
        self.params = list(self.nn1.parameters()) + list(self.nn2.parameters())
        self.optim2 = t.optim.SGD(self.params, 1e-3)
    
    def f1(self, x, d1):
        _x1 = self.nn1(x)+d1
        return _x1 

    def f2(self, x, d1, d2):
        _x1 = self.nn1(x)+d1
        _x2 = self.nn2(_x1)+d2
        return _x2 

    def loss(self, y_pred, y):
        return t.abs(y_pred-y).mean()

    def train(self, x, y1, y2, d1, d2):
        y_pred1 = self.f1(x, d1)
        loss = self.loss(y_pred1, y1)
        self.optim1.zero_grad()
        loss.backward()
        self.optim1.step()
        #print('\n\n optim1:\n', self.params)

        y_pred = self.f2(x, d1, d2)
        loss = self.loss(y_pred, y2)
        self.optim2.zero_grad()
        loss.backward()
        self.optim2.step()
        #print('\n\n optim2:\n', self.params)
        return loss

ns = 1000
nt = 100000
nn = NN()
# train
for i in range(nt):
    x = t.rand(ns,1)
    d1 = t.rand(ns,1)
    d2 = t.rand(ns,1)
    y1 = t.pow(x, 2)+d1
    y2 = t.pow(y1, 2)+d2
    loss = nn.train(x, y1, y2, d1, d2)
    print('loss: ', loss)
    #input()

#test
x = t.rand(ns,1)
d1 = t.rand(ns,1)
d2 = t.rand(ns,1)
y1 = t.pow(x, 2)+d1
y2 = t.pow(y1, 2)+d2
y_pred = nn.f2(x, d1, d2)
plt.figure()
plt.plot(x.detach().numpy(), y2.detach().numpy(), '.', label='gt')
plt.plot(x.detach().numpy(), y_pred.detach().numpy(), 'o', label='pred')
plt.show()




