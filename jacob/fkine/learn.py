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
from glob import glob
import json

def learn(models_dir, results_dir, plots_dir, model_kwargs, learn_kwargs, device):
    print('Learning Model: %s'%model_kwargs['model'])
    make_dirs([models_dir, results_dir, plots_dir])
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = "/fkine_"+_suffix
    
    kwargs_file = models_dir+fkine_file+'_kwargs.json'
    with open(kwargs_file, 'w') as f:
        json.dump(model_kwargs, f)

    if glob(models_dir+fkine_file+"*.pt"):
        print('models exist')
        if learn_kwargs['append']:
            with open(results_dir+fkine_file+'.pickle', 'rb') as h:
                losses, durations = pickle.load(h)
            #print('loading: ', losses, durations)
        else:
            print('no training')
            return
    else:
        losses = None 
        durations = []

    env_kwargs={'model_file':path.cwd()/('rgym/envs/assets/reacher%dd%dj.xml'%(model_kwargs['n_dims'], model_kwargs['n_joints']))}

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
    
    obs = dict()
    for key in envs.unwrapped.observation_space.keys():
        obs[key] = np.empty(shape=(envs.num_envs,)+envs.unwrapped.observation_space[key].shape)
    #a = np.zeros((envs.num_envs, envs.action_space.shape[0]))
    
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
            for env in envs.envs:
                _q = env.unwrapped.sample_joints() 
                env.unwrapped.set_joint_state(_q) 

            #obs, _, _, _ = envs.step(a)
            istep += envs.num_envs 
            run = model_cb._on_step()
        loss = model_cb._on_rollout_end(bs=learn_kwargs['batch_size'], n_iter=learn_kwargs['n_iter'])
        mean_losses.append(loss)
    durations.append(time() - start_time)

    if not isinstance(losses, np.ndarray):
        losses = np.array([mean_losses])
    else:
        losses = np.vstack((losses, np.array(mean_losses)))
    torch.save(fkine_net.state_dict(), models_dir+fkine_file+"run%d.pt"%losses.shape[0])

    plt.figure()
    epochs = np.linspace(1, losses.shape[1], losses.shape[1])
    plt.fill_between(epochs, np.min(losses, axis=0), np.max(losses, axis=0), alpha=0.3)
    plt.plot(epochs, np.mean(losses, axis=0))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(plots_dir+fkine_file+".png")
    with open(results_dir+fkine_file+'.pickle', 'wb') as h:
        pickle.dump((losses, durations), h)
    return
