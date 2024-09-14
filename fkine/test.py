import time, math, sys
import numpy as np
import torch
from pathlib import Path as path

from PIL import Image

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym # Utility stuff from utils import * 
from utils import *

if __name__ == "__main__":
    n_joints = 7
    n_dims = 3
    env = make_env(
            render_mode="rgb_array",
            model_file=path.cwd()/'rgym/envs/assets/reacher3d7j.xml',
            width=1440,
            height=720
            ) 
    obs = env.reset()
    n_images=100
    imgs = []
    qdot = np.zeros(n_joints+3,)
    for i in range(n_images):
        q = np.hstack((env.get_wrapper_attr('sample_joints')(), env.get_wrapper_attr('goal')))
        env.get_wrapper_attr('set_state')(q, qdot)
        arr = env.render()
        imgs.append(arr)
    
    ssdir = path.cwd()/'results/screeshots'
    ssdir.mkdir(parents=True, exist_ok=True)
    
    for i, img in enumerate(imgs):
        im = Image.fromarray(img)
        im.save(ssdir.as_posix()+'/ss%d.jpeg'%i)
   
    env.close()
