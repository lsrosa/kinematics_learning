import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from gymnasium import RewardWrapper

from rgym.envs.ex_reacher_v0 import ExReacherEnv

import os

class ReacherPolicy(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        
        # compute bounds for observation space
        _jl = self.unwrapped.joints_limits.max(axis=0)
        _max_xdot = self.unwrapped.qdot_max*self.unwrapped.link_radii.max() 
        _max_goal = (self.unwrapped.base_pos + self.unwrapped.link_radii.max()).max()
        _min_goal = (self.unwrapped.base_pos - self.unwrapped.link_radii.max()).min()
        
        self.observation_space = gym.spaces.Dict({
            'q': Box(low=_jl[0], high=_jl[1], shape=(self.unwrapped.n_joints,), dtype=np.float64),
            'xdot': Box(low=-_max_xdot, high=_max_xdot, shape=(self.unwrapped.n_dims,self.unwrapped.n_joints), dtype=np.float64),
            'goal': Box(low=_min_goal, high=_max_goal, shape=(self.unwrapped.n_dims,), dtype=np.float64)
            })

        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(self.unwrapped.n_joints,), dtype=np.float64)
        
        # this observation space is computed here and passed to the policy features extractor 
        self.policy_observation_space = gym.spaces.Dict({
            'x': Box(low=_min_goal, high=_max_goal, shape=(self.unwrapped.n_dims,self.unwrapped.n_joints), dtype=np.float64),
            'xdot': Box(low=-_max_xdot, high=_max_xdot, shape=(self.unwrapped.n_dims,self.unwrapped.n_joints), dtype=np.float64),
            'goal': Box(low=_min_goal, high=_max_goal, shape=(self.unwrapped.n_dims,), dtype=np.float64)
            })

        self.current_obs_x = None
        self.x_prev = None
        self.current_reward = None
        self.current_info = None
        self.current_obs = None

        return

    def step(
            self, action: ActType
        ):   #-> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_reward, self.current_info = self.reward(observation)
        
        # save current x value for the next velocity computation
        self.compute_observation()
        self.x_prev = self.current_obs_x

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
        self.current_obs = {'q': q, 'xdot': xdot, 'goal':self.unwrapped.goal}
        self.current_obs_x = _x
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
    
from gymnasium.envs.registration import register

def reacherpolicy(**args):
    env = gym.make("Ex-Reacher-v0", **args)
    env = ReacherPolicy(env)
    return env

register(
    id="ReacherPolicy",
    entry_point="rgym.envs.reacherPolicy:reacherpolicy",
    max_episode_steps=50,
    )

