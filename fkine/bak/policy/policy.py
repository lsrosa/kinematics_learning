import torch 
import torch.nn as nn
import itertools as it
import timeit

from typing import Callable
from gymnasium import spaces
from torch.utils.data import Dataset, DataLoader 
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.policies import ActorCriticPolicy

def create_NN(n_states, n_hidden_layers, activation, initializer):
    net = nn.Sequential()
    
    net.append(nn.Linear(n_states, n_hidden_layers[0]))
    net.append(activation())
    
    for n_in, n_out in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
        net.append(nn.Linear(n_in, n_out))
        net.append(activation())
    net.append(nn.Linear(n_hidden_layers[-1], n_states))
    
    for l in net.children():
        if isinstance(l, nn.Linear):
            initializer(l.weight)
            l.bias.data.fill_(0.0)
    return net

class Policy(nn.Module):
    def __init__(self, n_states, n_hidden_layers, activation, initializer, device):
        super(Policy, self).__init__()
        
        self.latent_dim_pi = n_states 
        self.latent_dim_vf = n_states     

        # save these for later use
        self.n_states = n_states
        self.h_dim = n_hidden_layers

        self.policy = create_NN(n_states, n_hidden_layers, activation, initializer) 
        self.value = create_NN(n_states, n_hidden_layers, activation, initializer) 
        return
    
    def set_fkine(self, fkine):
        self.fkine = fkine
        return

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy(features), self.value(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value(features)


class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        
        self.n_states = kwargs.pop('n_states')
        self.n_hidden_layers = kwargs.pop('n_hidden_layers')
        if 'activation' in kwargs:
            self.activation = kwargs.pop('activation')
        else:
            self.activation = nn.ReLU

        if 'initializer' in kwargs:
            self.initializer = kwargs.pop('initializer')
        else:
            self.initializer = nn.init.xavier_uniform
        self._device = kwargs.pop('device')

        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = Policy(self.n_states, self.n_hidden_layers, self.activation, self.initializer, self._device)
