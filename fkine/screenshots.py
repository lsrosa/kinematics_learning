import time, math, sys
import numpy as np
import torch
from pathlib import Path as path

from PIL import Image

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym # Utility stuff from utils import * 
from utils import *

if __name__ == "__main__":
    n_joints = 5
    n_dims = 3
    env = make_env(
            render_mode="rgb_array",
            model_file=path.cwd()/'rgym/envs/assets/marrtino.xml',
            width=2880,
            height=1440
            ) 
    obs = env.reset()
    
    qdot = np.zeros(n_joints+3,)
    q = np.array([np.pi/4, np.pi/2, np.pi/2, -np.pi/2, np.pi/4, 0., 0., 0.])
    env.get_wrapper_attr('set_state')(q, qdot)
    arr = env.render()
    
    ssdir = path.cwd()/'results/screeshots'
    ssdir.mkdir(parents=True, exist_ok=True)
    
    im = Image.fromarray(arr)
    im.save(ssdir.as_posix()+'/marrtino.jpeg')
   
    env.close()
