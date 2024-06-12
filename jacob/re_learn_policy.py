import torch
import gymnasium as gym

from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor 

from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono

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

class FKineWrap(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim):
        super().__init__(fkine, from_observation_space, to_observation_space)
        self.fkine = fkine 
        self.combine = CombinedExtractor(to_observation_space)
        return

    def forward(self, observations):
        out = dict()
        out['x'] = self.fkine(observations['q'])
        out['xdot'] = observations['xdot']
        out['goal'] = observations['goal']
        return self.flatten(out) 


# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def re_learn(models_dir, results_dir, plots_dir, env_kwargs, sac_kwargs, learn_kwargs, n_envs=32, device='cpu'):
    make_dirs([models_dir, results_dir, plots_dir])
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')
    print(env_kwargs['model_file']) 
    
    sac_model_file = models_dir+'/policy/%s.zip'%model_name
    print(sac_model_file)
    if not os.path.exists(sac_model_file):
        print('sac model does not exist, exiting')
        return

    fkine_model_files = sorted(list(path(models_dir+'/fkine_models').glob('*%s*.pt'%model_name.replace('reacher',''))))
    fkine_kwargs_files = sorted(list(path(models_dir+'/fkine_models').glob('*%s*.json'%model_name.replace('reacher',''))))
    if len(fkine_model_files) == 0:
        print('fkine models do not exist, exiting')
    
    for fkine_model_file, fkine_kwargs_file in zip(fkine_model_files, fkine_kwargs_files):
        print(fkine_model_file, fkine_kwargs_file)
        _env = make_vec_env("ReacherPolicy", env_kwargs=env_kwargs, n_envs=n_envs)
        env = make_vec_env("ReacherRePolicy", env_kwargs=env_kwargs, n_envs=n_envs)
        
        with open(fkine_kwargs_file, 'r') as f:
            fkine_kwargs = json.load(f)
        fkine_model = eval(fkine_kwargs.pop('model'))(**fkine_kwargs, device=device) 

        _n_features = 1
        for key in _env.observation_space.keys():
            for _l in _env.observation_space[key].shape:
                _n_features *= _l
        
        sac_kwargs['policy_kwargs'] =  {'features_extractor_class': FKineWrap,
                                        'features_extractor_kwargs': {'fkine' : fkine_model,
                                            'from_observation_space' : env.observation_space,
                                            'to_observation_space' : _env.observation_space}
                }
        print('loading policy model')
        sac_model = POLICY_TYPE.load(sac_model_file, env=_env, **sac_kwargs)
        sac_model.env = env
        print(sac_model.policy.features_extractor) 
        break 
    exit() 


    prev_timesteps = model._total_timesteps 

    # TODO: there is something weird were we have rollouts without training every so often, as result the reward has two curves 
    learn_kwargs['total_timesteps'] += prev_timesteps 
    model.learn(**learn_kwargs, log_interval=4, tb_log_name=model_name, reset_num_timesteps=False)
    
    model.save(models_dir+'/'+model_name)
    return

def play(models_dir, env_kwargs):
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')

    env_kwargs['render_mode'] = 'human'
    env = make_env(env_name='ReacherRePolicy', **env_kwargs)
    
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
    sac_hyperparams_dir = path.cwd()/'results/policy/tunning/'
    models_dir = path.cwd()/'rgym/envs/assets'

    env_kwargs = dict()
    for n_dims in [2]:#[2, 3]:
        for n_joints in [2]:#[2, 3, 4, 5, 6, 7]:
            models = sorted(models_dir.glob('reacher%dd%dj*.xml'%(n_dims, n_joints))) 
            for model in models:
                model_name = model.parts[-1].replace('.xml','')
                with open(sac_hyperparams_dir/model_name/'hyperparameters_1.json') as f:
                    sac_kwargs = json.load(f)
                    sac_kwargs['policy_kwargs'] = dict()
                    sac_kwargs['policy_kwargs']['log_std_init'] = sac_kwargs.pop('log_std_init')
                    sac_kwargs['policy_kwargs']['net_arch'] = net_arch[sac_kwargs.pop('net_arch')]
                    sac_kwargs['policy_kwargs']['use_sde'] = False 
                print('Loaded SAC parameters: ', sac_kwargs)
                env_kwargs['model_file'] = model
                re_learn('results', 'results/re_policy', 'results/re_policy', env_kwargs, sac_kwargs, learn_kwargs, device=device)
                continue
    #play('results/policy', env_kwargs)
