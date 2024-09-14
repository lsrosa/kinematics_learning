from pathlib import Path as path
import numpy as np

import gymnasium as gym
from rgym.envs.reachertest import ReacherTest
from rgym.envs.reacherPolicy import ReacherPolicy

from pathlib import Path as path
from dm_control import mjcf

def model_kwargs_2_str(n_dims, n_joints, lr, n_hidden, size_hidden, model=None):
    string = ''
    if model != None:
        string += model
    
    string += "_"+str(n_dims)+"d"+str(n_joints)+"j"+"_"+str(lr)+"_"+str(n_hidden)+"_"+str(size_hidden) 
    return string

def policy_kwargs_2_str(**kwargs):
    string = ''
    
    string += "_"+str(kwargs['n_dims'])+"d"+str(kwargs['n_joints'])+"j"
    return string

def make_env(env_name = "ReacherTest", **kwargs):
    env = gym.make(env_name, **kwargs)#, render_mode = 'human') 
    return env

def make_dirs(names = []):
    cwd = path.cwd()
    for name in names:
        (cwd/name).mkdir(parents=True, exist_ok=True)

def create_model_variation(model_dir, n_dims, n_joints, var):
    var_dir = path(model_dir)/'variations'
    path.mkdir(var_dir, exist_ok=True)
    model_name = 'reacher%dd%dj'%(n_dims, n_joints)
    
    var_number = len(sorted(var_dir.glob('reacher%dd%dj*.xml'%(n_dims, n_joints))))+1
    model_fullpath = model_dir+'/'+model_name+'.xml'
    print('\n modifying model %s'%model_fullpath)
    model = mjcf.from_path(model_fullpath)

    # Change link lenghts
    for j in range(n_joints):
        link = model.find('geom', 'link%d'%j)
        #joint = model.find('joint', 'j%d'%j)
        link.fromto = abs(np.random.normal(link.fromto, var*(link.fromto!=0))) 
        joint = model.find('body', 'j%d'%j)
        joint.pos = link.fromto[-3:]
    # change fingertip

    mjcf.export_with_assets(model, var_dir, model_name+'_var%d.xml'%var_number, precision=4)

    return

def create_models_variations():
    for n_dims in [3]:#[2, 3]:
        for n_joints in [7]:#[2, 3, 4, 5, 6, 7]:
            for delta in 0.01*np.ones(5):
                create_model_variation('rgym/envs/assets', n_dims, n_joints, delta)
    return

if __name__ == "__main__":
    create_models_variations()
