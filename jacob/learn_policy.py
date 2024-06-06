import torch
import gymnasium as gym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from utils import *

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def learn(models_dir, results_dir, plots_dir, model_kwargs, learn_kwargs, device):
    make_dirs([models_dir, results_dir, plots_dir])
    _suffix = policy_kwargs_2_str(**model_kwargs)
   
    if os.path.exists(models_dir+"/policy"+_suffix+".zip") and not learn_kwargs['refine']:
        print('model exist, no training')
        return
    
    env_kwargs={'n_dims':model_kwargs['n_dims'],
                'n_joints':model_kwargs['n_joints']
                }
    env = make_vec_env("ReacherPolicy", env_kwargs=env_kwargs, n_envs=learn_kwargs['n_envs'])

    if os.path.exists(models_dir+"/policy"+_suffix+".zip") and learn_kwargs['refine']:
        print('Loading existing model')
        model = SAC.load(models_dir+'/policy'+_suffix, env=env)
    else:
        print('Creating new model')
        model = SAC("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=models_dir+'/logs')
    # TODO: there is something weird were we have rollouts without training every so often, as result the reward has two curves 
    model.learn(total_timesteps=model._total_timesteps+learn_kwargs['total_timesteps'], log_interval=4, tb_log_name='model'+_suffix, reset_num_timesteps=False)
    
    model.save(models_dir+"/policy"+_suffix)
    return

def play(models_dir, model_kwargs):
    _suffix = policy_kwargs_2_str(**model_kwargs)
    model_kwargs['render_mode'] = 'human'
    env = make_env(env_name='ReacherPolicy', **model_kwargs)
    
    model = SAC.load(models_dir+'/policy'+_suffix)

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
    learn_kwargs['total_timesteps'] = 1e4
    learn_kwargs['n_envs'] = 32
    learn_kwargs['refine'] = True 
    
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:
            model_kwargs = dict()
            model_kwargs['n_joints'] = n_joints 
            model_kwargs['n_dims'] = n_dims 

            learn('policy', 'policy', 'policy', model_kwargs, learn_kwargs, device=device)

    #play('policy', model_kwargs)
