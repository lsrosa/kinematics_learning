import time, math, sys
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env

# Import ouur model
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from fkine.learn import learn
from fkine.fkine_common import get_hyper_params

# Utility stuff
from matplotlib import pyplot as plt
import seaborn, pandas, itertools
from utils import * 
import pickle
from glob import glob

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

if __name__ == '__main__':
    hyperparams_dir = path.cwd()/'results/tunning_fkine'
    model_kwargs_link = dict()
    model_kwargs_mono = model_kwargs_link.copy()

    learn_kwargs_link = dict()
    learn_kwargs_link['seed'] = 1
    learn_kwargs_link['n_rollouts'] = 100
    learn_kwargs_link['learn_steps'] = 5000 
    learn_kwargs_link['n_envs'] = 32 
    learn_kwargs_link['n_iter'] = 25 
    learn_kwargs_link['append'] = True 
    learn_kwargs_link['refine'] = False 
    learn_kwargs_mono = learn_kwargs_link.copy()
    
    n_runs = 50
    n_samples = 100

    for n_dims in [3]:#, 2]:
        for _nj, n_joints in enumerate([7, 6, 5, 4, 3, 2]):
            print('dims: ', n_dims, '   joints: ', n_joints)
            hp_file_link = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineLinked_hyperparams.pickle'%(n_dims, n_joints))))
            hp_file_mono = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineMono_hyperparams.pickle'%(n_dims, n_joints))))
            if (not hp_file_link):
                if not hp_file_link: print('could not find fkineLinked hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
                #continue
                
            if (not hp_file_mono):
                if not hp_file_mono: print('could not find fkineMono hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
                continue

            model_kwargs_link['model'] = 'FKineLinked'
            model_kwargs_link['n_dims'] = n_dims 
            model_kwargs_link['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_link[0])
            print('link: ', learn_params, model_params, n_params)
            model_kwargs_link.update(model_params)
            learn_kwargs_link.update(learn_params)
            for run in range(n_runs):
                learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, learn_kwargs_link, device=device)
    
            model_kwargs_mono['model'] = 'FKineMono'
            model_kwargs_mono['n_dims'] = n_dims 
            model_kwargs_mono['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_mono[0])
            print('mono: ', learn_params, model_params, n_params)
            model_kwargs_mono.update(model_params)
            learn_kwargs_mono.update(learn_params)
            for run in range(n_runs):
                learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_mono, learn_kwargs_mono, device=device)


