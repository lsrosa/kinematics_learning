import numpy as np
import torch
import torch.nn as nn
import os
from utils import *
import numpy as np

from stable_baselines3.common.env_util import make_vec_env

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


n_envs = 3
n_joints = 3
n_dims = 3
kwargs = {'n_dims':n_dims, 'n_joints':n_joints, 'render_mode':'human'}
envs = make_vec_env("ReacherTest", env_kwargs=kwargs, n_envs=n_envs)

ns=500
obs = envs.reset()

joints_limits = (envs.envs[0].unwrapped.model.jnt_range)[:n_joints]
qv = np.zeros(envs.envs[0].unwrapped.model.nv)

null_obs = dict()
for key in obs.keys():
    null_obs[key] = np.zeros(shape=obs[key].shape)

for i in range(10):
    _obs = []
    for i in range(n_envs):
        q = rand_between(joints_limits)
        qpos = np.hstack((q,np.zeros(3)))
        envs.envs[i].unwrapped.set_state(qpos, qv)
        _obs.append(envs.envs[i].get_observation())
    
    obs = null_obs.copy()
    for key in obs.keys():
        for i in range(len(_obs)):
            obs[key][i] = _obs[i][key]
    envs.render()
    input()
