import torch
from stable_baselines3.common.torch_layers import CombinedExtractor 
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
import gymnasium as gym
import json
import numpy as np
from gymnasium.spaces import Box
from pathlib import Path as path

class FKineWrap(CombinedExtractor):
    def __init__(self, from_observation_space, to_observation_space, **kwargs):
        n_joints = from_observation_space['q'].shape[0]
        n_dims = from_observation_space['goal'].shape[0]
        super().__init__(to_observation_space)

        device = kwargs['device'] 
        
        # we are not using freeze for now
        if 'freeze' in kwargs:
            freeze = kwargs['freeze']
        else:
            freeze = True
        
        # load fkine model kwargs
        if 'fkine_kwargs_file' in kwargs:
            fkine_kwargs_file = kwargs['fkine_kwargs_file']
        else: 
            fkine_kwargs_file = path.cwd()/'results/fkine_models/default_kwargs.json'

        with open(fkine_kwargs_file, 'r') as f:
            fkine_kwargs = json.load(f)                                     
        fkine_model_name = fkine_kwargs.pop('model')
        self.fkine = eval(fkine_model_name)(**fkine_kwargs, device=device)
        
        # Load pre-trained fkine model
        if 'fkine_model_file' in kwargs:
            fkine_model_file = kwargs['fkine_model_file']
        else:
            fkine_model_file = path.cwd()/'results/fkine_models/default_model.pt'
        self.fkine.load_state_dict(torch.load(fkine_model_file, map_location=torch.device(device)))
        
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