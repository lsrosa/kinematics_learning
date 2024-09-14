import time, math, sys
import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import itertools as it
from matplotlib import pyplot as plt

# Utility stuff
from utils import * 

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.nn = nn.Sequential()
        self.nn.append(nn.Linear(2, 3))
        self.nn.append(nn.Linear(3, 2))
    
        self.l = torch.tensor([0.5, 1.0], requires_grad=True)
        
        self.loss = nn.MSELoss(reduction='sum')
        self.params = dict()
        self.optim = torch.optim.SGD(list(self.nn.parameters())+[self.l], lr=1e-4)
    
    def state_dict(self):
        d = dict()
        d['nn'] = self.nn.state_dict()
        d['l'] = self.l
        return d

    def load_state_dict(self, d):
        self.nn.load_state_dict(d['nn'])
        self.l = d['l']
        return

    def forward(self, xin):
        return self.nn(xin) + self.l


if __name__ == "__main__":
    ns = 100
    nn1 = NN()
    for i in range(1000):
        x = torch.rand(ns, 2)
        y = x**2 + torch.tensor([2, 1])
        
        nn1.optim.zero_grad()
        y_pred = nn1(x)
        loss = nn1.loss(y, y_pred)
        loss.backward()
        nn1.optim.step()

    print(nn1.l)
    print('saving\n', nn1.state_dict())
    torch.save(nn1.state_dict(), 'test.pt')

    nn2 = NN()
    print('before loading\n\n', nn2.state_dict())
    nn2.load_state_dict(torch.load('test.pt'))
    print(' after loading\n\n', nn2.state_dict())

    plt.figure()
    plt.plot(y.detach().numpy())
    plt.plot(y_pred.detach().numpy())
    plt.show()
    input()
