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
import json
from pathlib import Path as path

def learn(models_dir, results_dir, plots_dir, model_kwargs, learn_kwargs, env_kwargs=None, device='cpu', plot=False, noisy=False, noise_var=0.0):
    model_kwargs = model_kwargs.copy()
    models_dir = path(models_dir)
    results_dir = path(results_dir)
    plots_dir = path(plots_dir)

    if 'env_models_home' in model_kwargs:
        env_models_home = path(model_kwargs.pop('env_models_home'))
    else:
        env_models_home = path.cwd()
    
    print('Learning Model: %s'%model_kwargs['model'])
    make_dirs([models_dir, results_dir, plots_dir])
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = "fkine_"+_suffix
    if noisy: fkine_file = fkine_file + '_var%f'%noise_var
    
    kwargs_file = models_dir/(fkine_file+'_kwargs.json')
    with open(kwargs_file, 'w') as f:
        json.dump(model_kwargs, f)
    
    fkine_model_files = sorted(list(models_dir.glob("%s*.pt"%fkine_file)))
    print('model_files: ', fkine_model_files)

    if len(fkine_model_files)>0:
        print('Existing models: ', fkine_model_files)
        if learn_kwargs['append'] and learn_kwargs['refine']:
            print('Choose between refine or append. Doing nothing.')
            return

        if learn_kwargs['append'] or learn_kwargs['refine']:
            with open(results_dir/(fkine_file+'.pickle'), 'rb') as h:
                losses, durations = pickle.load(h)
                steps_learned = losses.shape[1]

            if learn_kwargs['append']:
                fkine_model_files = [models_dir/(fkine_file+"run%d.pt"%(losses.shape[0]+1))]
                steps_to_learn = steps_learned
            elif learn_kwargs['refine']:
                steps_to_learn = learn_kwargs['learn_steps']-steps_learned
                if steps_to_learn <= 0:
                    print('Already learned more than learn_spes. No training.')
                    return
                print('already learned: ', steps_learned, ' ... to learn: ', steps_to_learn)
        else:
            print('no training')
            return
    else:
        steps_learned = 0 
        steps_to_learn = learn_kwargs['learn_steps']
        fkine_model_files.append(models_dir/(fkine_file+"run1.pt"))
        losses = np.empty(shape=(0,steps_to_learn)) 
        durations = np.empty(shape=(0,))

    if env_kwargs == None: 
        env_kwargs={'model_file':env_models_home/('rgym/envs/assets/reacher%dd%dj.xml'%(model_kwargs['n_dims'], model_kwargs['n_joints']))}
    print(env_kwargs)

    envs = make_vec_env("ReacherTest", env_kwargs=env_kwargs, n_envs=learn_kwargs['n_envs']) 
    print("----------------------------")
    print(f"Obs: {envs.observation_space}   Act: {envs.action_space}")
    
    learn_steps = steps_to_learn*envs.num_envs*learn_kwargs['n_rollouts']
    current_seed = steps_learned*envs.num_envs

    mean_losses = np.empty(shape=(0,steps_to_learn))
    
    fkine_model_name = model_kwargs.pop('model')
    for fkine_model_file in fkine_model_files:
        _mean_losses = []
        _durs = []
        
        # train
        fkine_net = eval(fkine_model_name)(**model_kwargs, device=device)
        if learn_kwargs['refine'] and losses.shape[0]>0:
            print('loading for continuing training: ', fkine_model_file)
            fkine_net.load_state_dict(torch.load(fkine_model_file, map_location=torch.device(device)))        
        fkine_net.to(device)

        model_cb = LearnCB(device = device)
        model_cb.env = envs 
        model_cb.fkine = fkine_net
        model_cb.data = IncrDataset() 
        
        obs = dict()
        for key in envs.unwrapped.observation_space.keys():
            obs[key] = np.empty(shape=(envs.num_envs,)+envs.unwrapped.observation_space[key].shape)
        
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
                run = model_cb._on_step(noisy, noise_var)
            loss = model_cb._on_rollout_end(bs=learn_kwargs['batch_size'], n_iter=learn_kwargs['n_iter'])
            _mean_losses.append(loss)
        _durs.append(time() - start_time)
        
        mean_losses = np.vstack((mean_losses, np.array(_mean_losses)))
        
        # Save model
        torch.save(fkine_net.state_dict(), fkine_model_file)
    
    # append or add durations and losses
    _durs = np.array(_durs)
    if learn_kwargs['refine']:
        if losses.shape[0] == 0:
            losses = mean_losses
            durations = _durs
        else:
            losses = np.hstack((losses, mean_losses))
            durations = durations + _durs
    else:
        losses = np.vstack((losses, mean_losses))
        durations = np.hstack((durations, _durs))

    # Save durations and losses
    with open(results_dir/(fkine_file+'.pickle'), 'wb') as h:
        pickle.dump((losses, durations), h)
    
    if not plot: return
    # make a pretty figure
    plt.figure()
    epochs = np.linspace(1, losses.shape[1], losses.shape[1])
    plt.fill_between(epochs, np.min(losses, axis=0), np.max(losses, axis=0), alpha=0.3)
    plt.plot(epochs, np.mean(losses, axis=0))
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(plots_dir/(fkine_file+'.png'))

    return
