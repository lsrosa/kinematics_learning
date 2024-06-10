import time, math, sys
import numpy as np
import torch
from pathlib import Path as path

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym # Utility stuff from utils import * 
from utils import *

if __name__ == "__main__":
    n_joints = 2
    n_dims = 2
    
    model_file = path.cwd()/'rgym/envs/assets/reacher3d6j.xml'

    envs = make_env('ReacherPolicy', render_mode="human", model_file=model_file) 
    obs = envs.reset()
    input()
    for j in reversed(range(n_joints)):
        print('j%d'%j)
        a = np.zeros(n_joints)
        for i in range(10):
            a[j] = 0.1
            obs, r, _, _, _ = envs.step(a)
            print(obs, r)
            envs.render()
        input()
    
