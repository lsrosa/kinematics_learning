import torch 
import torch.nn as nn
import itertools as it
import timeit

from torch.utils.data import Dataset, DataLoader 
from stable_baselines3.common.callbacks import BaseCallback

import numpy as np
from pathlib import Path as path
import pandas
from paretoset import paretoset

#------------- Read and choose hyperparams ------------
def get_hyper_params(hp_file):
    metrics = pandas.read_pickle(hp_file).filter(['loss', 'learn_steps', 'num_params'])
    configs = pandas.read_pickle(hp_file).filter(['config/lr', 'config/nh', 'config/sh', 'config/batch_size'])

    mask = paretoset(metrics, sense=['min', 'min', 'min'])
    best_configs = configs.get(mask)
    best_metrics = metrics.get(mask)
    
    print('Best Configurations:\n', best_configs)
    print('Best Metrics:\n', best_metrics)

    idx = best_metrics['loss'].idxmin()
    _best_config = best_configs.loc[idx].to_dict()
    
    # update dictionary keys
    best_config = dict((k.replace('config/','').replace('nh', 'n_hidden').replace('sh', 'size_hidden'), v) for (k, v) in _best_config.items())
    
    learn_config = dict((k, int(best_config[k]) if k in ['batch_size'] else best_config[k]) for k in ['batch_size'])
    model_config = dict((k, int(best_config[k]) if k in ['n_hidden', 'size_hidden'] else best_config[k]) for k in ['lr', 'n_hidden', 'size_hidden'])

    return learn_config, model_config, best_metrics['num_params'].loc[idx] 

#------------------- Incremental Dataset --------------
class IncrDataset(Dataset):
    def __init__(self, size=0, max_size=1e5, dropout_size=None):
        self.size = size
        self.max_size = max_size
        if dropout_size == None:
            self.dropout = round(max_size/10)
        else:
            self.dropout = dropout_size
        
        self.q = []
        self.qdot = []
        self.x = []
        self.xdot = []
    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        q = self.q[idx]
        qdot = self.qdot[idx]
        x = self.x[idx]
        xdot = self.xdot[idx]
        return q, qdot, x, xdot
    
    # TODO: should try to pre-allocate the memory and use indexes instead of allocating and de-allocating memory
    def add(self, q, qdot, x, xdot):
        if self.size == self.max_size:
            self.q = self.q[self.dropout:]
            self.qdot = self.qdot[self.dropout:]
            self.x = self.x[self.dropout:]
            self.xdot = self.xdot[self.dropout:]
            self.size = len(self.q)
        else:
            self.size += 1

        self.q.append(torch.tensor(q, dtype=torch.float32))
        self.qdot.append(torch.tensor(qdot, dtype=torch.float32))
        self.x.append(torch.tensor(x, dtype=torch.float32))
        self.xdot.append(torch.tensor(xdot, dtype=torch.float32))

# ------------------- ModelLearnCB -------------------

class LearnCB(BaseCallback):
    def __init__(self, device = 'cpu', user_quit = [False], verbose: int = 0):
            super().__init__(verbose)
            self.user_quit = user_quit
            self.fkine = None
            self.env = None
            self.data = None 
            self.device = device
            self.count_rollouts = 0

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self, noisy=False, noise_var=0.0) -> bool:
        if self.fkine != None:
            for env in self.env.envs:
                obs = env.compute_observation()  # using wrappers
                rew, info = env.get_reward()

                # add noise to obs
                if noisy:
                    obs['x'] = obs['x']+np.random.normal(0.0, noise_var, size=obs['x'].shape)

                self.data.add(obs['q'], obs['qdot'], obs['x'], obs['xdot'])
        return not self.user_quit[0]


    def _on_rollout_end(self, bs, n_iter) -> None:
        loss = None
        self.count_rollouts += 1
        
        data_loader = DataLoader(dataset=self.data, batch_size=bs, shuffle=True, generator=torch.Generator(device='cpu'))
        
        if self.fkine != None: 
            loss = 0
        
        for t in range(n_iter):
            q, qdot, x, xdot = next(iter(data_loader))
            q = q.to(self.device)
            y = x.to(self.device)

            if self.fkine != None:
                loss += self.fkine.train_from_data(q, y)
            
        if self.fkine != None:
            loss /= n_iter
        
        print(f" loss mean_loss {loss} | iter {n_iter} ")
        return loss 

    def _on_training_end(self) -> None:

        pass


