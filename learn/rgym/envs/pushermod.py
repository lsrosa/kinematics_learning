import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from gymnasium import RewardWrapper

from rgym.envs.pusher_v4 import PusherEnv

from networks import *


class PusherFK(gym.Wrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.current_observation = None
        self.current_reward = None
        self.current_info = None
        self.fknet = None

    def set_fknet(self, fknet):
        self.fknet = fknet

    def reset(self,*,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
        obs, info = super().reset(seed=seed)
        self.current_observation = obs
        a,b = self.env.reset()
        
        x0 = self.unwrapped.data.qpos.flat[0:7]       # joints
        obj = self.unwrapped.get_body_com("object")   # object pos
        goal = self.unwrapped.get_body_com("goal")    # goal pos
        
        self.xgoal = norm_pi(self.fknet.inverse(x0, goal, tol=0.01, min_grad=1e-5, iters=1000, verbose=0).detach().numpy())
        self.xobj = norm_pi(self.fknet.inverse(x0, obj, tol=0.01, min_grad=1e-5, iters=1000, verbose=0).detach().numpy())
        # print(f" -- reset | t {vstr(t)} | xgoal {vstr(self.xgoal)}")
        return self.get_observation(),info

    def step(
            self, action: ActType
        ):   #-> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Modifies the :attr:`env` :meth:`step` reward using :meth:`self.reward`."""
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.current_observation = observation
        self.current_reward = reward
        self.current_info = info
        return observation, reward, terminated, truncated, info

    def get_observation(self) -> WrapperObsType:
        return self.current_observation

    def get_reward(self):
        return self.current_reward, self.current_info



class PusherFK21ad(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
        a = norm_pi(self.unwrapped.data.qpos.flat[0:7]-self.xgoal[0:7])   # goal angle diff
        b = norm_pi(self.unwrapped.data.qpos.flat[0:7]-self.xobj[0:7])    # object angle diff
        return np.concatenate(
            [
                self.unwrapped.data.qvel.flat[:7],  # angle vel (observation[7:14])
                a,      # goal angle diff
                b       # object angle diff
            ] )


class PusherFK37ad(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(23+14,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
        a = norm_pi(self.unwrapped.data.qpos.flat[0:7]-self.xgoal[0:7])   # goal angle diff
        b = norm_pi(self.unwrapped.data.qpos.flat[0:7]-self.xobj[0:7])    # object angle diff
        return np.concatenate(
            [
                observation,
                a,      # goal angle diff
                b       # object angle diff
            ] )


def pusherfk21ad(**args):
    env = gym.make("Pusher-v4b", **args)
    env = PusherFK(env)
    env = PusherFK21ad(env)
    return env


def pusherfk37ad(**args):
    env = gym.make("Pusher-v4b", **args)
    env = PusherFK(env)
    env = PusherFK37ad(env)
    return env

    
from gymnasium.envs.registration import register


register(
     id="Pusher-v4b",
     entry_point="rgym.envs.pusher_v4:PusherEnv",
     max_episode_steps=100,
)


register(
     id="PusherFK21ad",
     entry_point="rgym.envs.pushermod:pusherfk21ad",
     max_episode_steps=100,
)

register(
     id="PusherFK37ad",
     entry_point="rgym.envs.pushermod:pusherfk37ad",
     max_episode_steps=100,
)


