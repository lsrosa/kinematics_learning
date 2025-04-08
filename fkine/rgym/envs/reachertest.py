import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium import RewardWrapper

from rgym.envs.ex_reacher_v0 import ExReacherEnv

import os

class ReacherTest(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        
        self.observation_space = gym.spaces.Dict({
            'q': Box(low=-np.inf, high=np.inf, shape=(self.unwrapped.n_joints,), dtype=np.float64),
            'qdot': Box(low=-np.inf, high=np.inf, shape=(self.unwrapped.n_joints,), dtype=np.float64),
            'x': Box(low=-np.inf, high=np.inf, shape=(self.unwrapped.n_dims,self.unwrapped.n_joints), dtype=np.float64),
            'xdot': Box(low=-np.inf, high=np.inf, shape=(self.unwrapped.n_dims,self.unwrapped.n_joints), dtype=np.float64)
            })
        
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.unwrapped.n_joints,), dtype=np.float64)
        
        self.x_prev = None
        self.current_reward = None
        self.current_info = None
        self.current_obs = None

        return

    def step(
            self, action: ActType
        ):   #-> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        
        self.current_reward, self.current_info = self.reward(observation)
        
        # save current x value for the next velocity computation
        self.compute_observation()
        self.x_prev = self.current_obs['x']

        return self.current_obs, self.current_reward, self.current_info['goal_achieved'], truncated, self.current_info 
    
    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        
        # position of each joint
        _x = np.zeros((self.unwrapped.n_dims, self.unwrapped.n_joints))
        
        for j in range(self.unwrapped.n_joints):
            _x[:, j] = self.unwrapped.get_body_com('j%d'%j)[:self.unwrapped.n_dims]
        self.x_prev = _x 
        
        self.compute_observation()

        return self.get_observation(), info

    def get_observation(self) -> WrapperObsType:
        return self.current_obs

    def compute_observation(self) -> WrapperObsType:
        # joint angles
        q = self.unwrapped.data.qpos[:self.unwrapped.n_joints]
        qdot = self.unwrapped.data.qvel.flat[:self.unwrapped.n_joints]           # joint ang vel
        
        # position of each joint
        _x = np.zeros((self.unwrapped.n_dims, self.unwrapped.n_joints))
        for j in range(self.unwrapped.n_joints):
            _x[:, j] = self.unwrapped.get_body_com('j%d'%j)[0:self.unwrapped.n_dims]

        xdot = (_x - self.x_prev)/self.unwrapped.dt

        self.current_obs = {'q': q, 'qdot': qdot, 'x': _x, 'xdot': xdot}
        return self.current_obs 

    def get_reward(self):
        return self.current_reward, self.current_info

    def reward(self, observation):
        dist_ee_target = observation[-self.unwrapped.n_dims:]
        r_dist = np.linalg.norm(dist_ee_target)
        vel = self.current_obs['xdot'][:,-1]
        r_vel = np.linalg.norm(vel)
       
        g = r_dist<0.05 and r_vel<0.1
        
        info = dict()
        info['goal_achieved'] = g
        info['reward_vel'] = r_vel     
        info['reward_dist'] = r_dist 

        return r_dist + r_vel, info 
    
    def sample_states(self, n_samples, strategy='walk'):
        q, qdot, x, xdot = [], [], [], []
        
        for sample in range(n_samples):
            if strategy == 'walk': 
                action = self.action_space.sample() 
                obs, _, _, _, _ = self.step(action)
            elif strategy == 'random': 
                _q = self.unwrapped.sample_joints() 
                self.unwrapped.set_joint_state(_q) 
                obs = self.compute_observation() 
            
            q.append(obs['q'].copy())
            qdot.append(obs['qdot'].copy())
            x.append(obs['x'].copy())
            xdot.append(obs['xdot'].copy())
        
        q = np.array(q)
        qdot = np.array(qdot)
        x = np.array(x)
        xdot = np.array(xdot)
        return q, qdot, x, xdot


from gymnasium.envs.registration import register

def reachertest(**args):
    env = gym.make("Ex-Reacher-v0", **args)
    env = ReacherTest(env)
    return env

register(
     id="ReacherTest",
     entry_point="rgym.envs.reachertest:reachertest",
     max_episode_steps=50,
)

