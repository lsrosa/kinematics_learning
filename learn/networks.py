# -*- coding: utf-8 -*-
# Neural networks

import math
import numpy as np
import timeit

from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple, Type, Union, SupportsFloat

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchdata.datapipes as dp

import gymnasium as gym

from stable_baselines3.common.callbacks import BaseCallback

device = "cuda" if torch.cuda.is_available() else "cpu"
device = 'cpu'

DEBUG=0


def check_norm_pi(a):
    if type(a)==torch.Tensor and len(a.shape)>0:
        for i in range(a.shape[0]):
            check_norm_pi(a[i])
    else:
        assert a<=math.pi and a>=-math.pi, f"Angle {a} not normalized"

def norm_pi(a):   # [-PI, PI)
    if (type(a)==torch.Tensor or type(a)==np.ndarray) and len(a.shape)>0:
        for i in range(a.shape[0]):
            try:
                a[i] = norm_pi(a[i])
            except Exception as e:
                # print(e)
                check_norm_pi(a[i])
    else:
        if a>=math.pi:
            a -= 2*math.pi
        elif a<-math.pi:
            a += 2*math.pi
    return a


def vstr(x):
    if type(x)==float:
        r = f"{x:7.4f}"
    elif type(x)!=list and len(x.shape)==0:
        r = f"{x:7.4f}"
    else:
        r = ""
        for i,_ in enumerate(x):
            r += f"{float(x[i]):7.4f} "
    return r



# Incremental Dataset 
class IncrDataset(Dataset):
    def __init__(self, size=0):
        self.size = size
        self.collected = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.collected[idx]

    def add(self,t):
        self.collected.append(torch.tensor(t, dtype=torch.float32))
        self.size += 1




# ---------------------------- FKNet -------------------------

class FKNet(torch.nn.Module):

    def __init__(self, n_in=2, n_out=2):
        super().__init__()
        self.n_out = n_out
        self.n_in = n_in

        h = 256
        self.fmodel = nn.Sequential(
            nn.Linear(self.n_in, h),
            nn.ReLU(),
            nn.Linear(h, 32),
            nn.ReLU(),
            nn.Linear(32, self.n_out*16),
            nn.Tanh(),
            nn.Linear(self.n_out*16, self.n_out)
        )

    def forward(self, x):
        x = norm_pi(x)
        y = self.fmodel(x)
        return y

    def predict(self, x):
        if type(x) is not torch.Tensor:
            x = torch.Tensor(x)
        assert type(x) == torch.Tensor
        y = self.forward(x)
        return y

    def print_summary(self):
        print("--------------------------")
        print(self.fmodel[0].weight.dtype)
        print(f"Input: {(self.n_in, )}")
        for l in self.fmodel:
            print(l)
        print(f"Output: {(self.n_out, )}")
        train_nparams = sum(p.numel() for p in self.fmodel.parameters() if p.requires_grad)

        print(f"Trainable parameters {train_nparams}")
        print("--------------------------")


    def derivative(self, obs, grad0=None):

        with torch.enable_grad():

            if type(obs)==torch.Tensor:
                x = obs.clone().detach().requires_grad_(True)
            else:
                # print(type(obs))
                x = torch.tensor(obs).type(torch.float32).requires_grad_(True)

            y = self.forward(x)

            if grad0 is None:
                grad0 = torch.ones_like(y)   # same shape as ytest
            elif type(grad0)!=torch.Tensor:
                grad0 = torch.tensor(grad0).type(torch.float32)

            # grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grad0, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
            # print(grad)

            y.backward (gradient = grad0, retain_graph = True)

            # print(x.grad.size())

            grad = x.grad

        return grad


    def inverse(self, x0, t, tol=0.01, min_grad=1e-5, iters=1000, verbose=0):
        '''
        Local Inverse FK
            x0: start angles
            t: target pos
            tol: distance tolerance
            Return: y: target angles
        '''
        if type(x0)!=torch.Tensor:
            x0 = torch.tensor(x0).type(torch.float32)
        if type(t)!=torch.Tensor:
            t = torch.tensor(t).type(torch.float32)

        eta = 0.5
        m = 0
        eta2 = 1
        d = 1
        nm = 1
        x = x0
        x = norm_pi(x)
        p = self.forward(x)
        i = 0
        while d>tol and i<iters and nm > min_grad:
            g = self.derivative(x,t-p)
            m = np.clip(eta2 * m + eta * g, -0.1, 0.1)
            nm = torch.norm(m)
            x += m
            x = norm_pi(x)
            p = self.forward(x)
            d = torch.norm(p-t)
            i += 1
            if verbose>0:
                print(f"   -- x {vstr(x)} | g {vstr(g)} | m {vstr(m)} | p {vstr(p)} | t-p {vstr(t-p)} | d {vstr(d)}")

        #if d>tol:
        #    print(f"   inverse -- x {vstr(x)} | g {vstr(g)} | m {vstr(m)} | p {vstr(p)} | t-p {vstr(t-p)} | d {vstr(d)}")

        return x

    def train(self, mode):  # called by RL alg
        # no training during RL learn
        return

    def train_from_data(self, data, niter=1000):
        global log_writer, log_iter

        dl = DataLoader(dataset=data, batch_size=30, shuffle=True,
                generator=torch.Generator(device=device))

        loss_fn = nn.MSELoss(reduction='sum')

        learning_rate = 1e-4

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        start = timeit.default_timer()
        t0 = start

        mean_loss = 0

        for t in range(niter):

            d = next(iter(dl))

            x = d[:,0:-self.n_out]           # x
            y = d[:,-self.n_out:].view(-1,self.n_out)  # f(x)

            y_pred = self.forward(x)

            loss = loss_fn(y_pred,y)
            mean_loss += loss.item()

            '''
            log_writer.add_scalar("train/loss", loss, train_steps)
            log_writer.flush()
            '''

            if DEBUG>0 and t % 100 == 0:
                t1 = timeit.default_timer()
                print(f"     | loss {loss.item():.6f} | time {t1-t0:.2f}")
                t0 = timeit.default_timer()

            # Zero the gradients before running the backward pass.
            optimizer.zero_grad()

            # Backward pass: compute gradient of the loss with respect to all the learnable
            # parameters of the model. Internally, the parameters of each Module are stored
            # in Tensors with requires_grad=True, so this call will compute gradients for
            # all learnable parameters in the model.
            loss.backward()

            # Update the weights
            optimizer.step()

        end = timeit.default_timer()

        mean_loss /= niter
        print(f"  mean_loss {mean_loss:.6f} | iter {niter} | time {end-start:.2f}")


    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        state = torch.load(filename)
        self.load_state_dict(state)





# ---------------------------- ModelLearnCB -------------------------


class ModelLearnCB(BaseCallback):

    def __init__(self, user_quit = [False], verbose: int = 0):
            super().__init__(verbose)
            self.user_quit = user_quit
            self.fknet = None
            self.env = None
            self.fkdata = IncrDataset()
            self.count_rollouts = 0
            print('ModelLearnCB')

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """

        if self.fknet != None:

            for env in self.env.envs:
                obs = env.unwrapped.get_observation()  # using wrappers
                #rew, info = env.unwrapped.get_reward()
                
                #print(obs)
                #print(rew)
                #print(info)

                if env.spec.id[0:10] == 'AbsReacher':
                    y = env.unwrapped.xpos
                    self.fkdata.add((obs[0],obs[1],y[0],y[1]))
                elif env.spec.id[0:6] == 'Pusher':
                    y = env.unwrapped.get_body_com("tips_arm")
                    self.fkdata.add((obs[0],obs[1],obs[2],obs[3],obs[4],obs[5],obs[6],y[0],y[1],y[2]))
                #elif env.spec.id[0:7] == 'Reacher':
                #    TODO 
                else:
                    assert False, f"Unknown environment {env.spec.id} !!!"

                # print(f"fkdata {obs} {y}")

                

        return not self.user_quit[0]


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.count_rollouts += 1

        #if self.count_rollouts%100==0:

        if self.fknet != None:
            # print(f"ModelLearnCB: rollouts {self.count_rollouts}")
            self.fknet.train_from_data(self.fkdata, niter=1000)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
                
        pass


