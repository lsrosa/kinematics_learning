import numpy as np
import torch
import torch.nn as nn
import os

n = 32
net = nn.Sequential()
net.append(nn.Linear(4, n)) #q0, q1, q2, q3
net.append(nn.ReLU())
net.append(nn.Linear(n, n))
net.append(nn.ReLU())
net.append(nn.Linear(n, n))
net.append(nn.ReLU())
net.append(nn.Linear(n, n))
net.append(nn.ReLU())
net.append(nn.Linear(n, 3)) #x, y, z

ns=500
if not os.path.exists('test.pt'):
    r = torch.tensor([1.1, 1.2, 1.3, 1.4])
    loss = nn.MSELoss()
    optim = torch.optim.SGD(net.parameters(), lr=1e-4)
    for step in range(500):
        q = (torch.rand(ns, 4)*2*torch.pi)
        q_acc = q.cumsum(dim=1)
        x = (r*torch.cos(q_acc)).sum(dim=1)
        y = (r*torch.sin(q_acc)).sum(dim=1)
        z = torch.zeros(ns)
        label = torch.vstack((x, y, z)).t()
        y_pred = net(q)
        l = loss(label, y_pred)
        print(l)
        optim.zero_grad()
        l.backward()
        optim.step()

    torch.save(net.state_dict(), 'test.pt')


if os.path.exists('test.pt'):
    net.load_state_dict(torch.load('test.pt'))

    q = (torch.rand(4)*2*torch.pi).requires_grad_(True)
    j = torch.autograd.functional.jacobian(net, q)
    print('jacob : ', j)

