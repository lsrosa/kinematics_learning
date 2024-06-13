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

class FKineLinkedWrap(CombinedExtractor):
    def __init__(self, from_observation_space, to_observation_space, fkine):
        super().__init__(to_observation_space)
        self.fkine = fkine
        return

    def forward(self, observations):
        out = dict()
        with torch.no_grad():
            out['x'], _ = self.fkine(observations['q'])
        out['xdot'] = observations['xdot']
        out['goal'] = observations['goal']
        return super().forward(out) 

class FKineMonoWrap(CombinedExtractor):
    def __init__(self, from_observation_space, to_observation_space, fkine):
        super().__init__(to_observation_space)
        self.fkine = fkine
        return

    def forward(self, observations):
        out = dict()
        with torch.no_grad():
            out['x'] = self.fkine(observations['q'])
        out['xdot'] = observations['xdot']
        out['goal'] = observations['goal']

        return super().forward(out) 

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def re_learn(models_dir, results_dir, plots_dir, env_kwargs, sac_kwargs, learn_kwargs, refine=False, n_envs=32, device='cpu'):
    make_dirs([models_dir, results_dir, plots_dir])
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')
    print('Env model: ', env_kwargs['model_file']) 
    
    total_ts = learn_kwargs['total_timesteps'] 

    fkine_model_files = sorted(list(path(models_dir+'/fkine_models').glob('*%s*.pt'%model_name.replace('reacher',''))))
    fkine_kwargs_files = sorted(list(path(models_dir+'/fkine_models').glob('*%s*.json'%model_name.replace('reacher',''))))
    if len(fkine_model_files) == 0:
        print('fkine models do not exist, exiting')
    
    env = make_vec_env("ReacherRePolicy", env_kwargs=env_kwargs, n_envs=n_envs)
    _env = make_vec_env("ReacherPolicy", env_kwargs=env_kwargs, n_envs=n_envs)
        
    for fkine_model_file, fkine_kwargs_file in zip(fkine_model_files, fkine_kwargs_files):
        env.reset()
        
        # load the pre-trained fkine
        with open(fkine_kwargs_file, 'r') as f:
            fkine_kwargs = json.load(f)
            
        fkine_model_name = fkine_kwargs.pop('model') 
        
        sac_model_name = model_name + fkine_model_name 
        sac_model_file = models_dir+'/re_policy/%s.zip'%sac_model_name

        if os.path.exists(sac_model_file) and not refine:
            print('sac model exist, no training')
            return
        fkine_model = eval(fkine_model_name)(**fkine_kwargs, device=device) 

        sac_kwargs['policy_kwargs']['features_extractor_class'] = eval(fkine_model_name+'Wrap')
        sac_kwargs['policy_kwargs']['features_extractor_kwargs'] = {
                'fkine': fkine_model,
                'to_observation_space': _env.observation_space
                }

        if os.path.exists(sac_model_file) and refine:
            print('Loading existing model')
            sac_model = POLICY_TYPE.load(sac_model_file, env=env, verbose=1)
            prev_timesteps = sac_model._total_timesteps 
        else:
            print('Creating new model')
            sac_model = POLICY_TYPE("MultiInputPolicy", env, verbose=1, device=device, tensorboard_log=models_dir+'/re_policy/logs', **sac_kwargs)
            prev_timesteps = 0 

        # TODO: there is something weird were we have rollouts without training every so often, as result the reward has two curves 
        learn_kwargs['total_timesteps'] = total_ts + prev_timesteps
        sac_model.learn(**learn_kwargs, log_interval=4, tb_log_name=sac_model_name, reset_num_timesteps=False)

        sac_model.save(sac_model_file)
    return

def play(models_dir, env_kwargs):
    model_name = env_kwargs['model_file'].parts[-1].replace('.xml','')

    env_kwargs['render_mode'] = 'human'
    env = make_env(env_name='ReacherRePolicy', **env_kwargs)
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
    learn_kwargs['total_timesteps'] = 1e5
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
                re_learn('results', 'results/re_policy', 'results/re_policy', env_kwargs, sac_kwargs, learn_kwargs, refine=True, device=device)
                continue
    play('results/re_policy', env_kwargs)
