import torch
from stable_baselines3.common.torch_layers import CombinedExtractor 
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
import gymnasium as gym
import json
import numpy as np
from gymnasium.spaces import Box

def get_policy_observation_space(n_joints, n_dims):
    pos = gym.spaces.Dict({
        'x': Box(low=-np.inf, high=np.inf, shape=(n_dims,n_joints), dtype=np.float64),
        'xdot': Box(low=-np.inf, high=np.inf, shape=(n_dims,n_joints), dtype=np.float64),
        'goal': Box(low=-np.inf, high=np.inf, shape=(n_dims,), dtype=np.float64)
        })
    return pos

class FKineWrap(CombinedExtractor):
    def __init__(self, from_observation_space, **kwargs):
        n_joints = from_observation_space['q'].shape[0]
        n_dims = from_observation_space['goal'].shape[0]
        super().__init__(get_policy_observation_space(n_joints, n_dims))

        device = kwargs['device'] 
        if 'freeze' in kwargs:
            freeze = kwargs['freeze']
        else:
            freeze = True

        fkine_kwargs_file = kwargs['fkine_kwargs_file']
        with open(fkine_kwargs_file, 'r') as f:
            fkine_kwargs = json.load(f)                                     
        fkine_model_name = fkine_kwargs.pop('model')
        self.fkine = eval(fkine_model_name)(**fkine_kwargs, device=device)
        if 'fkine_model_file' in kwargs:
            self.fkine.load_state_dict(torch.load(kwargs['fkine_model_file']))
        
        self.forward = eval('self.forward%s'%fkine_model_name)
        return

    def forwardFKineLinked(self, observations):
        out = dict()
        #TODO: check this no grad
        with torch.no_grad():
            out['x'], _ = self.fkine(observations['q'])
        out['xdot'] = observations['xdot']
        out['goal'] = observations['goal']
        return super().forward(out) 
    
    def forwardFKineMono(self, observations):
        out = dict()
        #TODO: check this no grad
        with torch.no_grad():
            out['x'] = self.fkine(observations['q'])
        out['xdot'] = observations['xdot']
        out['goal'] = observations['goal']
        return super().forward(out) 
