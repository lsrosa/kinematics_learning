import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env

from utils import *

MODEL_TYPE = SAC

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
        model = MODEL_TYPE.load(models_dir+'/'+model_name, env=env, **sac_kwargs)
        prev_timesteps = model._total_timesteps 
    else:
        print('Creating new model')
        model = MODEL_TYPE("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=models_dir+'/logs', **sac_kwargs)
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
    
    model = MODEL_TYPE.load(models_dir+'/'+model_name)

    for i in range(10):
        obs, info = env.reset()
        for s in range(100):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                continue
    return

if __name__ == '__main__':
    learn_kwargs = dict()
    learn_kwargs['total_timesteps'] = 1e3

    sac_kwargs = dict()
    sac_kwargs['learning_rate'] = 1e-4
    
    models_dir = Path.cwd()/'rgym/envs/assets/variations'

    env_kwargs = dict()
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:
            models = sorted(models_dir.glob('reacher%dd%dj*.xml'%(n_dims, n_joints))) 
            for model in models:
                env_kwargs['n_joints'] = n_joints 
                env_kwargs['n_dims'] = n_dims 
                env_kwargs['model_file'] = model
                learn('policy', 'policy', 'policy', env_kwargs, sac_kwargs, learn_kwargs, refine=False, device=device)
                continue
    play('policy', env_kwargs)
