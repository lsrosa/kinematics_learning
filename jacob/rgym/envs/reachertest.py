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

'''
Reacher 
- https://gymnasium.farama.org/environments/mujoco/reacher/
- https://github.com/openai/gym/blob/master/gym/envs/mujoco/reacher_v4.py

Action Space        Box(-1.0, 1.0, (2,), float32)

                    torque applied to each joint

Observation Space   Box(-inf, inf, (11,), float64)

0 cosine of the angle of the first arm
1 cosine of the angle of the second arm
2 sine of the angle of the first arm
3 sine of the angle of the second arm
4 x-coordinate of the target            position (m)
5 y-coordinate of the target            position (m)
6 angular velocity of the first arm     angular velocity (rad/s)
7 angular velocity of the second arm    angular velocity (rad/s)
8 x-value of position_fingertip - position_target        position (m)
9 y-value of position_fingertip - position_target        position (m)
10 z-value of position_fingertip (0)    position (m)


XML file

0 angle of the first arm        angle (rad)
1 angle of the second arm       angle (rad)
2 x-coordinate of the target    position (m)
3 y-coordinate of the target    position (m)


reward_distance: This reward is a measure of how far the fingertip of the reacher (the unattached end) is from the target, with a more negative value assigned for when the reacher’s fingertip is further away from the target. It is calculated as the negative vector norm of (position of the fingertip - position of target), or -norm(“fingertip” - “target”).

reward_control: A negative reward for penalising the walker if it takes actions that are too large. It is measured as the negative squared Euclidean norm of the action, i.e. as - sum(action2).

The total reward returned is reward = reward_distance + reward_control

The episode ends when any of the following happens:

Truncation: The episode duration reaches a 50 timesteps (with a new random target popping up if the reacher’s fingertip reaches it before 50 timesteps)

Termination: Any of the state space values is no longer finite.

'''

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
        q = np.mod(self.unwrapped.data.qpos[:self.unwrapped.n_joints], 2*np.pi)
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

    def reward(self, observation, reward, info):
        dist_ee_target = observation[-self.unwrapped.n_dims:]
        r_dist = np.linalg.norm(dist_ee_target)
        vel = self.current_obs['xdot'][:,-1]
        r_vel = np.linalg.norm(vel)
       
        g = r_dist<0.05 and r_vel<0.1
        
        info = Dict()
        info['goal_achieved'] = g
        info['reward_vel'] = r_vel     
        info['reward_dist'] = r_dist 

        return r_dist + r_vel, info 
    
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

