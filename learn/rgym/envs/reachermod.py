import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.envs.mujoco import ReacherEnv
from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.wrappers import FrameStack
from gymnasium import RewardWrapper

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

class ReacherMod(gym.Wrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.current_observation = None
        self.current_reward = None
        self.current_info = None

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
        

class ReacherMod2(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
       
        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
            ] )


class ReacherMod4(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
       
        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
                self.unwrapped.get_body_com("target")[0:2], # target
            ] )


class ReacherMod6v(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
       
        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
                self.unwrapped.get_body_com("target")[0:2], # target
                self.unwrapped.data.qvel.flat[:2]           # joint ang vel
            ] )


class ReacherMod6d(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:
       
        d = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")

        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
                self.unwrapped.get_body_com("target")[0:2], # target
                d[:2]                                       # tip - target pos
            ] )


class ReacherMod8(gym.ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:

        d = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")
       
        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
                self.unwrapped.get_body_com("target")[0:2], # target
                self.unwrapped.data.qvel.flat[:2],          # joint ang vel
                d[:2]                                       # tip - target pos
            ] )


class ReacherModSR(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs,info = super().reset(**kwargs)
        return obs,info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(observation,reward,info), terminated, truncated, info

    def reward(self,observation, reward, info):
        p1 = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")
        d1 = np.linalg.norm(p1)
        v1 = self.unwrapped.data.qvel.flat[:2]
        vn1 = np.linalg.norm(v1)

        evn1 = 2/(1 + math.exp(vn1/10))-1      # [0,40]  -> [0,-1]     
        ed1 = 2/(1 + math.exp(d1/0.1))-1       # [0,0.4] -> [0,-1]

        g = 1.0 if (d1<0.05 and vn1<1.0) else 0.0

        rv = - ed1 * evn1
        rv2 = ed1 if ed1 < -0.5 else rv

        info['reward_goal'] = g
        info['reward_vel'] = evn1     
        info['reward_dist'] = ed1
        # info['reward_ctrl'] = np.linalg.norm(action)

        return rv2 + g




from gymnasium.envs.registration import register


def reachermod2(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod2(env)
    return env


def reachermod4(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod4(env)
    return env

def reachermod4x2(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod4(env)
    env = FrameStack(env,2)
    return env

def reachermod4x2sr(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod4(env)
    env = FrameStack(env,2)
    env = ReacherModSR(env)
    return env

def reachermod6v(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod6v(env)
    return env

def reachermod6vsr(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    return env


def reachermod6d(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod6d(env)
    return env

def reachermod8(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherMod8(env)
    return env



register(
     id="ReacherMod2",
     entry_point="rgym.envs.reachermod:reachermod2",
     max_episode_steps=100,
)

register(
     id="ReacherMod4",
     entry_point="rgym.envs.reachermod:reachermod4",
     max_episode_steps=100,
)

register(
     id="ReacherMod4x2",
     entry_point="rgym.envs.reachermod:reachermod4x2",
     max_episode_steps=100,
)

register(
     id="ReacherMod4x2SR",
     entry_point="rgym.envs.reachermod:reachermod4x2sr",
     max_episode_steps=100,
)

register(
     id="ReacherMod6v",
     entry_point="rgym.envs.reachermod:reachermod6v",
     max_episode_steps=100,
)

register(
     id="ReacherMod6vSR",
     entry_point="rgym.envs.reachermod:reachermod6vsr",
     max_episode_steps=100,
)

register(
     id="ReacherMod6d",
     entry_point="rgym.envs.reachermod:reachermod6d",
     max_episode_steps=100,
)

register(
     id="ReacherMod8",
     entry_point="rgym.envs.reachermod:reachermod8",
     max_episode_steps=100,
)



