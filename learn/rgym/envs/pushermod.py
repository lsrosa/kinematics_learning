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





from gymnasium.envs.registration import register


register(
     id="Pusher-v4b",
     entry_point="rgym.envs.pusher_v4:PusherEnv",
     max_episode_steps=100,
)


