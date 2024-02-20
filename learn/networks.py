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


DEBUG=0

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


    def derivative(self, obs, grd=None):

        with torch.enable_grad():

            if type(obs)==torch.Tensor:
                x = obs.clone().detach().requires_grad_(True)
            else:
                # print(type(obs))
                x = torch.tensor(obs).type(torch.float32).requires_grad_(True)

            y = self.forward(x)

            if grd is None:
                grd = torch.ones_like(y)   # same shape as ytest
            elif type(grd)!=torch.Tensor:
                grd = torch.tensor(grd).type(torch.float32)

            # grad = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=grd, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=None, is_grads_batched=False, materialize_grads=False)
            # print(grad)

            y.backward (gradient = grd, retain_graph = True)

            # print(x.grad.size())

            grad = x.grad
 
        return grad


    def train(self, mode):  # called by RL alg
        # no training during RL learn
        return

    def train_from_data(self, data, niter=1000):

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
                rew, info = env.unwrapped.get_reward()
                
                #print(obs)
                #print(rew)
                #print(info)

                if env.spec.id[0:10] == 'AbsReacher':
                    y = env.unwrapped.xpos
                elif env.spec.id[0:7] == 'Reacher':
                    pass
                else:
                    assert False, "Unknown environment !!!"

                # print(f"fkdata {obs} {y}")

                self.fkdata.add((obs[0],obs[1],y[0],y[1]))

        return not self.user_quit[0]


    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        self.count_rollouts += 1

        if self.count_rollouts%100==0:
            print(f"ModelLearnCB: rollouts {self.count_rollouts}")

            if self.fknet != None:
                self.fknet.train_from_data(self.fkdata, niter=1000)


    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
                
        pass


