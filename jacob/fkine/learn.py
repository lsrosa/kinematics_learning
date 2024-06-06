import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env

# Import our model
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from fkine.fkine_common import LearnCB, IncrDataset

# Utility stuff
from matplotlib import pyplot as plt
from utils import * 
from time import time
import pickle

def learn(models_dir, results_dir, plots_dir, model_kwargs, learn_kwargs, device):
    print('Learning Model: %s'%model_kwargs['model'])
    make_dirs([models_dir, results_dir, plots_dir])
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = "/fkine_"+_suffix
    
    if os.path.exists(models_dir+fkine_file+".pt"):
        print('models exist, no training')
        return 

    env_kwargs={'n_dims':model_kwargs['n_dims'],
                'n_joints':model_kwargs['n_joints']}
    envs = make_vec_env("ReacherTest", env_kwargs=env_kwargs, n_envs=learn_kwargs['n_envs']) 
    print("----------------------------")
    print(f"Obs: {envs.observation_space}   Act: {envs.action_space}")
   
    # train
    fkine_net = eval(model_kwargs.pop('model'))(**model_kwargs, device=device)
    fkine_net.to(device)
    
    model_cb = LearnCB(device = device)
    model_cb.env = envs 
    model_cb.fkine = fkine_net
    model_cb.data = IncrDataset() 
    
    learn_steps = learn_kwargs['learn_steps']*envs.num_envs*learn_kwargs['n_rollouts']
    current_seed = learn_kwargs['seed'] 
    mean_losses = []
    a = np.zeros((envs.num_envs, envs.action_space.shape[0]))
    istep = 0
    run = True
    
    start_time = time()
    while istep < learn_steps and run:
        print(f"{istep:6d} | Training fkine_net with seed {current_seed} ...")
        envs.seed(seed=current_seed)
        current_seed += envs.num_envs
        obs = envs.reset()
        
        model_cb._on_rollout_start()
        for _ in range(learn_kwargs['n_rollouts']):
            for ienv in range(envs.num_envs):
                a[ienv] = envs.envs[ienv].action_space.sample()
            obs, _, _, _ = envs.step(a)
            istep += envs.num_envs 
            run = model_cb._on_step()
        losses = model_cb._on_rollout_end(bs=learn_kwargs['batch_size'], n_iter=learn_kwargs['n_iter'])
        mean_losses.append(losses)
    duration = time() - start_time

    losses = np.array(mean_losses) 
    torch.save(fkine_net.state_dict(), models_dir+fkine_file+".pt")

    plt.figure()
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(plots_dir+fkine_file+".png")
    with open(results_dir+fkine_file+'.pickle', 'wb') as h:
        pickle.dump((losses, duration), h)
    return
