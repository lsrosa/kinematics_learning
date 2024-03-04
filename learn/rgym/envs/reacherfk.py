import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from gymnasium import RewardWrapper

from rgym.envs.reacher_v4 import ReacherEnv

from networks import *


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

class ReacherFK(gym.Wrapper):

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
        x0 = self.unwrapped.data.qpos.flat[0:2]
        t = self.unwrapped.get_body_com("target")[0:2]
        #x0 = obs[0:2]
        #t = obs[2:4]
        self.xgoal = norm_pi(self.fknet.inverse(x0, t, tol=0.01, min_grad=1e-5, iters=1000, verbose=0).detach().numpy())
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



class ReacherFK4a(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
        a = norm_pi(self.unwrapped.data.qpos.flat[0:2]-self.xgoal[0:2])   # angle diff
        return np.concatenate(
            [
                a,                                      # joint angle diff
                self.unwrapped.data.qvel.flat[:2]       # joint ang vel
            ] )



class ReacherFK15(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(15,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
        a = norm_pi(self.unwrapped.data.qpos.flat[0:2]-self.xgoal[0:2])   # angle diff
        return np.concatenate(
            [
                observation,
                a,                                      # joint angle diff
                self.unwrapped.data.qvel.flat[:2]       # joint ang vel
            ] )


from rgym.envs.reachermod import ReacherModSR

from gymnasium.envs.registration import register

def reacherfk4a(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherFK(env)
    env = ReacherFK4a(env)
    env = ReacherModSR(env)
    return env

def reacherfk15(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherFK(env)
    env = ReacherFK15(env)
    env = ReacherModSR(env)
    return env

register(
     id="ReacherFK4aSR",
     entry_point="rgym.envs.reacherfk:reacherfk4a",
)

register(
     id="ReacherFK15SR",
     entry_point="rgym.envs.reacherfk:reacherfk15",
)

