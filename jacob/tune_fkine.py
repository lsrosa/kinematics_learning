import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env

# Import our model
from fkine.learn import learn

# Utility stuff
from matplotlib import pyplot as plt
from utils import * 

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

if __name__ == '__main__':
    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 100
    learn_kwargs['learn_steps'] = 300 
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['batch_size'] = 10 
    learn_kwargs['n_iter'] = 10 
    learn_kwargs['append'] = False
    learn_kwargs['refine'] = False
    
    out_dir = 'results/tunning_fkine'
    total = 2*1*2*2*2*7
    curr = 1
    for model in ['FKineLinked', 'FKineMono']:
        for lr in [1e-4]:#, 1e-5]:
            for nh in [5, 6]:
                for sh in [32, 64]:
                    for n_dims in [2, 3]:
                        for n_joints in [2, 3, 4, 5, 6, 7]:
                            print('curr: %d, out of %d'%(curr, total))
                            model_kwargs = dict()
                            model_kwargs['model'] = model 
                            model_kwargs['lr'] = lr
                            model_kwargs['n_hidden'] = nh
                            model_kwargs['size_hidden'] = sh 
                            model_kwargs['n_joints'] = n_joints 
                            model_kwargs['n_dims'] = n_dims 

                            learn(out_dir, out_dir, out_dir, model_kwargs, learn_kwargs, device=device) 
                            curr += 1
