import time, math, sys
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym # Utility stuff from utils import * 
from utils import *

if __name__ == "__main__":
    n_joints = 7
    n_dims = 3
    envs = make_env(render_mode="human", n_dims = n_dims, n_joints = n_joints) 
    obs = envs.reset()
    input()
    for j in reversed(range(n_joints)):
        print('j%d'%j)
        a = np.zeros(n_joints)
        for i in range(10):
            a[j] = 0.1
            print(a)
            envs.step(a)
            envs.render()
        input()
    
