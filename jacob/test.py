from utils import *
from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
import json
from re_learn_policy import FKineWrap
import torch

net_arch = {
        "t2": [32, 32],
        "t3": [32, 32, 32],
        "t4": [32, 32, 32, 32],
        "s2": [64, 64],
        "s3": [64, 64, 64],
        "s4": [64, 64, 64, 64],
    }
base_dir = '/home/leandro/repos/iros2024/jacob/'
with open('results/fkine_models/fkine_FKineLinked_2d2j_0.0005_3_32_kwargs.json', 'r') as f:
    fkine_kwargs = json.load(f)

fkine = FKineLinked(**fkine_kwargs)

env_kwargs = {'model_file': base_dir+'rgym/envs/assets/reacher2d2j.xml'}
env = make_vec_env('ReacherPolicy', env_kwargs=env_kwargs)
nenv = make_vec_env('ReacherRePolicy', env_kwargs=env_kwargs)

with open('results/policy/tunning/reacher2d2j/hyperparameters_1.json') as f:
    sac_kwargs = json.load(f)
    sac_kwargs['policy_kwargs'] = dict()
    sac_kwargs['policy_kwargs']['log_std_init'] = sac_kwargs.pop('log_std_init')
    sac_kwargs['policy_kwargs']['net_arch'] = net_arch[sac_kwargs.pop('net_arch')]
    sac_kwargs['policy_kwargs']['use_sde'] = False 


sac_model_file = base_dir+'results/policy/reacher2d2j.zip'
sac1 = SAC.load(sac_model_file, env = env, **sac_kwargs)

sac_kwargs['policy_kwargs']['features_extractor_class'] = FKineWrap
sac_kwargs['policy_kwargs']['features_extractor_kwargs'] = {
            'fkine': fkine,
            'from_observation_space': nenv.observation_space
            }
sac2 = SAC('MultiInputPolicy', nenv, **sac_kwargs)

sac.env = nenv
sac.policy.features_extractor_class = FKineWrap
sac.policy.features_extractor_kwargs = fe_kwargs


