import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env

from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from policy.wraps import FKineWrap

from utils import *
import json

POLICY_TYPE = SAC
net_arch = {
        "t4": [32 for i in range(4)],
        "t8": [32 for i in range(8)],
        "s4": [64 for i in range(4)],
        "s8": [64 for i in range(8)],
        "m4": [128 for i in range(4)],
        "m8": [128 for i in range(8)],
        "l4": [256 for i in range(4)],
        "l8": [256 for i in range(8)],
    }

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def learn(models_dir, results_dir, plots_dir, env_kwargs, sac_kwargs, learn_kwargs, refine=False, n_envs=32, device='cpu', verbose=0):
    make_dirs([models_dir, results_dir, plots_dir])
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')
    print('Env model: ', env_kwargs['model_file']) 
    
    env = make_vec_env("ReacherPolicy", env_kwargs=env_kwargs, n_envs=n_envs)
    
    # add the policy observation space to kwargs
    sac_kwargs['policy_kwargs']['features_extractor_kwargs']['to_observation_space'] = env.envs[0].get_wrapper_attr('policy_observation_space') 

    sac_model_name = model_name + sac_kwargs.pop('fkine_model_name')
    sac_model_file = models_dir+'/policy/%s.zip'%sac_model_name
    print(sac_model_file, sac_model_name)
    if os.path.exists(sac_model_file) and not refine:
        print('sac model exist, no training')
        return

    if os.path.exists(sac_model_file) and refine:
        print('Loading existing policy model')
        sac_model = POLICY_TYPE.load(sac_model_file, env=env, verbose=verbose)
    else:
        print('Creating new policy model')
        sac_model = POLICY_TYPE("MultiInputPolicy", env, verbose=verbose, device=device, tensorboard_log=models_dir+'/policy/logs', **sac_kwargs)

    # TODO: there is something weird were we have rollouts without training every so often, as result the reward has two curves 
    sac_model.learn(**learn_kwargs, log_interval=4, tb_log_name=sac_model_name, reset_num_timesteps=False)

    sac_model.save(sac_model_file)
    return

def play(models_dir, env_kwargs):
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')

    env_kwargs['render_mode'] = 'human'
    env = make_env(env_name='ReacherPolicy', **env_kwargs)
    for sac_model_name in ['reacher2d2jFKineLinked.zip', 'reacher2d2jFKineMono.zip']:
        env.reset()  
        model = POLICY_TYPE.load(models_dir+'/'+sac_model_name)

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
    sac_hyperparams_dir = path.cwd()/'results/policy/tunning/'
    models_dir = path.cwd()/'rgym/envs/assets'
    learn_kwargs['total_timesteps'] = 1e5
 
    env_kwargs = dict()
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:
            models = sorted(models_dir.glob('reacher%dd%dj*.xml'%(n_dims, n_joints))) 
            print(models)
            for model in models:
                model_name = model.parts[-1].replace('.xml','')
                print(model_name)
                
                # For each Fkine model 
                fkine_model_files = sorted(list(path('results/fkine_models').glob('*%s*.pt'%model_name.replace('reacher',''))))
                fkine_kwargs_files = sorted(list(path('results/fkine_models').glob('*%s*.json'%model_name.replace('reacher',''))))
                if len(fkine_model_files) == 0:
                    print('fkine models do not exist, exiting')

                for fkine_model_file, fkine_kwargs_file in zip(fkine_model_files, fkine_kwargs_files):
                    # load the pre-trained fkine
                    print('fkine model file: ', fkine_model_file)
                    print('fkine kwargs file: ', fkine_kwargs_file)
                    with open(fkine_kwargs_file, 'r') as f:
                        fkine_kwargs = json.load(f)
                    
                    fkine_model_name = fkine_kwargs['model']
                    with open(sac_hyperparams_dir/(model_name+fkine_model_name)/'hyperparameters_1.json') as f:
                        sac_kwargs = json.load(f)
                        sac_kwargs['policy_kwargs'] = dict()
                        sac_kwargs['policy_kwargs']['log_std_init'] = sac_kwargs.pop('log_std_init')
                        sac_kwargs['policy_kwargs']['net_arch'] = net_arch[sac_kwargs.pop('net_arch')]
                        sac_kwargs['policy_kwargs']['use_sde'] = False 
                        sac_kwargs['fkine_model_name'] = fkine_model_name 
                        sac_kwargs['policy_kwargs']['features_extractor_class'] = FKineWrap
                        sac_kwargs['policy_kwargs']['features_extractor_kwargs'] = {
                                'fkine_model_file': fkine_model_file,
                                'fkine_kwargs_file': fkine_kwargs_file,
                                'device': device,
                                'freeze': True
                                } 
                    print('Loaded SAC parameters: ', sac_kwargs)
                    env_kwargs['model_file'] = model
                    
                    learn('results', 'results/policy', 'results/policy', env_kwargs, sac_kwargs, learn_kwargs, refine=True, device=device, verbose=False)
                continue
    #play('results/policy', env_kwargs)
