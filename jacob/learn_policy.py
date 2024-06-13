import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env

from utils import *
import json

POLICY_TYPE = SAC

net_arch = {
        "t2": [32, 32],
        "t3": [32, 32, 32],
        "t4": [32, 32, 32, 32],
        "s2": [64, 64],
        "s3": [64, 64, 64],
        "s4": [64, 64, 64, 64],
    }

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def learn(models_dir, results_dir, plots_dir, env_kwargs, sac_kwargs, learn_kwargs, refine=False, n_envs=32, device='cpu'):
    make_dirs([models_dir, results_dir, plots_dir])
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')
     
    if os.path.exists(models_dir+'/'+model_name+".zip") and not refine:
        print('model exist, no training')
        return
     
    env = make_vec_env("ReacherPolicy", env_kwargs=env_kwargs, n_envs=n_envs)

    if os.path.exists(models_dir+'/'+model_name+".zip") and refine:
        print('Loading existing model')
        model = POLICY_TYPE.load(models_dir+'/'+model_name, env=env, verbose=1)
        prev_timesteps = model._total_timesteps 
    else:
        print('Creating new model')
        model = POLICY_TYPE("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=models_dir+'/logs', **sac_kwargs)
        prev_timesteps = 0 
    # TODO: there is something weird were we have rollouts without training every so often, as result the reward has two curves 
    learn_kwargs['total_timesteps'] += prev_timesteps 
    model.learn(**learn_kwargs, log_interval=4, tb_log_name=model_name, reset_num_timesteps=False)
    
    model.save(models_dir+'/'+model_name)
    return

def play(models_dir, env_kwargs):
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')

    env_kwargs['render_mode'] = 'human'
    env = make_env(env_name='ReacherPolicy', **env_kwargs)
    
    model = POLICY_TYPE.load(models_dir+'/'+model_name)

    for i in range(5):
        obs, info = env.reset()
        for s in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                continue
    return

if __name__ == '__main__':
    learn_kwargs = dict()
    learn_kwargs['total_timesteps'] = 1e6
    hyperparams_dir = path.cwd()/'results/policy/tunning/'
    models_dir = path.cwd()/'rgym/envs/assets'

    env_kwargs = dict()
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:
            models = sorted(models_dir.glob('reacher%dd%dj*.xml'%(n_dims, n_joints))) 
            for model in models:
                model_name = model.parts[-1].replace('.xml','')
                with open(hyperparams_dir/model_name/'hyperparameters_1.json') as f:
                    sac_kwargs = json.load(f)
                    sac_kwargs['policy_kwargs'] = dict()
                    sac_kwargs['policy_kwargs']['log_std_init'] = sac_kwargs.pop('log_std_init')
                    sac_kwargs['policy_kwargs']['net_arch'] = net_arch[sac_kwargs.pop('net_arch')]
                    sac_kwargs['policy_kwargs']['use_sde'] = False 
                print('Loaded SAC parameters: ', sac_kwargs)
                env_kwargs['model_file'] = model
                learn('results/policy', 'results/policy', 'results/policy', env_kwargs, sac_kwargs, learn_kwargs, refine=True, device=device)
                continue
    play('results/policy', env_kwargs)
