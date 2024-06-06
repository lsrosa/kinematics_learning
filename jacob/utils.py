import os
import numpy as np

import gymnasium as gym
from rgym.envs.reachertest import ReacherTest
from rgym.envs.reacherPolicy import ReacherPolicy

def model_kwargs_2_str(n_dims, n_joints, lr, n_hidden, size_hidden, model=None):
    string = ''
    if model != None:
        string += model
    
    string += "_"+str(n_dims)+"d"+str(n_joints)+"j"+"_"+str(lr)+"_"+str(n_hidden)+"_"+str(size_hidden) 
    return string

def policy_kwargs_2_str(n_dims, n_joints):
    string = ''
    
    string += "_"+str(n_dims)+"d"+str(n_joints)+"j"
    return string

def make_env(env_name = "ReacherTest", **kwargs):
    env = gym.make(env_name, **kwargs)#, render_mode = 'human') 
    return env

def make_dirs(names = []):
    for name in names:
        if not os.path.exists(name):
            os.makedirs(name)

def rand_between(limits):
    n = len(limits)
    ret = np.zeros(n)

    for i in range(n):
        low = limits[i,0]
        high = limits[i,1]

        ret[i] = np.random.rand()*(high-low)+low
    return ret
