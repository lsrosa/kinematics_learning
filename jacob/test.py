from utils import *
from stable_baselines3 import PPO, SAC 
from stable_baselines3.common.env_util import make_vec_env
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, CombinedExtractor 
import json
import torch

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

class FKineWrap(CombinedExtractor):
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

fkine = FKineLinked(**fkine_kwargs, device=device)
print('at instance: ', fkine.device)
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
sac1 = SAC.load(sac_model_file, env = env, **sac_kwargs, tensorboard_log='temp/', device=device)

sac_kwargs['policy_kwargs']['features_extractor_class'] = FKineWrap
sac_kwargs['policy_kwargs']['features_extractor_kwargs'] = {
            'fkine': fkine,
            'to_observation_space': env.observation_space
            }
sac2 = SAC('MultiInputPolicy', nenv, **sac_kwargs, tensorboard_log='temp/', device=device)

#sac1.learn(100, log_interval=4, tb_log_name='sac1')
print('\n\n before')
print(sac2.get_parameters()['policy']['actor.features_extractor.fkine.fkine0.fkine.0.weight'])
sac2.learn(1000, log_interval=4, tb_log_name='sac2')
print('\n\n  after')
print(sac2.get_parameters()['policy']['actor.features_extractor.fkine.fkine0.fkine.0.weight'])


