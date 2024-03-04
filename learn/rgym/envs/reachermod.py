import gymnasium as gym
import time, math, sys
import numpy as np
import torch

from typing import TYPE_CHECKING, SupportsFloat, Any, TypeVar

from gymnasium.core import ActType, ObsType, WrapperObsType
from gymnasium.spaces import Box
from gymnasium.wrappers import FrameStack
from gymnasium import RewardWrapper, ObservationWrapper

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


class ReacherMod2(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:

        return np.concatenate(
            [
                self.unwrapped.data.qpos.flat[:2],          # joint angles
            ] )


class ReacherMod4(ObservationWrapper):

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





class ReacherMod6v(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:

        return np.concatenate(
            [
                norm_pi(self.unwrapped.data.qpos.flat[:2]), # joint angles
                self.unwrapped.get_body_com("target")[0:2], # target
                self.unwrapped.data.qvel.flat[:2]           # joint ang vel
            ] )


class ReacherMod6d(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:

        d = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")

        return np.concatenate(
            [
                norm_pi(self.unwrapped.data.qpos.flat[:2]),          # joint angles
                d[0:2],                                     # tip - target pos
                self.unwrapped.data.qvel.flat[:2]           # joint ang vel
            ] )



class ReacherMod4d(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)

    def observation(self, observation: ObsType) -> WrapperObsType:

        d = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")

        return np.concatenate(
            [
                # norm_pi(self.unwrapped.data.qpos.flat[:2]),          # joint angles
                d[0:2],                                     # tip - target pos
                self.unwrapped.data.qvel.flat[:2]           # joint ang vel
            ] )


class ReacherMod6a(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)
        self.unwrapped.set_angle_target(True)

    def observation(self, observation: ObsType) -> WrapperObsType:

        return np.concatenate(
            [
                norm_pi(self.unwrapped.data.qpos.flat[:2]), # joint angles
                self.unwrapped.xgoal,                       # target angles
                self.unwrapped.data.qvel.flat[:2]           # joint ang vel
            ] )


class ReacherMod4a(ObservationWrapper):

    def __init__(self, env: gym.Env[ObsType, ActType]):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float64)
        self.unwrapped.set_angle_target(True)

    def observation(self, observation: ObsType) -> WrapperObsType:

        a = norm_pi(self.unwrapped.data.qpos.flat[0:2]-self.unwrapped.xgoal[0:2])   # angle diff


        return np.concatenate(
            [
                a,                                      # joint angle diff
                self.unwrapped.data.qvel.flat[:2]       # joint ang vel
            ] )


class ReacherMod8(ObservationWrapper):

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


def calc_reward(d1, v1, info):

    dn1 = np.clip(1-d1/0.4, 0, 1)          # [0,0.4] -> [1,0]
    vn1 = np.clip(1-v1/12, 0, 1)           # [0,12]  -> [1,0]

    vn1d = np.clip(math.pow(vn1,2*dn1), 0, 1)

    g = 1.0 if (d1<0.05 and v1<1.0) else 0.0

    rv = dn1 * vn1d

    rv2 = dn1 if dn1 < 0.5 else rv

    info['reward_goal'] = g
    info['reward_vel'] = vn1d
    info['reward_dist'] = dn1
    # info['reward_ctrl'] = np.linalg.norm(action)

    return rv + g


class ReacherModSR(RewardWrapper):

    def __init__(self, env):
        super().__init__(env)
        self.gamma = 0.99

    def reset(self, **kwargs):
        obs,info = super().reset(**kwargs)
        p0 = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")
        self.d0 = np.linalg.norm(p0)
        return obs,info

    def step(self, action):
        # print(f" === action {action[0]} ===")
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, self.reward(observation,action,reward,info), terminated, truncated, info

    def reward(self, observation, action, reward, info):
        pos1 = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")
        vel1 = self.unwrapped.data.qvel.flat[:2]
        d1 = np.linalg.norm(pos1)
        v1 = np.linalg.norm(vel1)

        r = calc_reward(d1, v1, info)
        return r



# fixed target
class ReacherModFT(gym.Wrapper):

    def __init__(self, env, target=None, mode=None):
        super().__init__(env)
        self.target = target
        self.mode = mode

    def set_fixed_target(self, t):
        self.target = t

    # set a fixed target
    def fix_target(self):
        if self.mode == 'A1':
            a = self.np_random.uniform(low=0, high=2*math.pi)
            self.target = [ 0.2*math.cos(a), 0.2*math.sin(a) ]
            # print(self.target)
        elif self.target == None:
            self.target = [0, 0]
        self.env.unwrapped.goal = np.array(self.target)
        qpos = self.env.unwrapped.data.qpos
        qvel = self.env.unwrapped.data.qvel
        qpos[-2:] = self.env.unwrapped.goal
        self.env.unwrapped.set_state(qpos, qvel)
        return self.env.unwrapped.get_observation()

    def reset(self, **kwargs):
        obs,info = super().reset(**kwargs)
        obs = self.fix_target()
        # print(f"Fixed target pos {self.unwrapped.get_body_com('target')}")
        return obs,info

    def step(self, action):
        return self.env.step(action)



# 1 dof
class ReacherModA1(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.action_space = Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        eaction = np.concatenate( [ action, np.array([0]) ] )
        return self.env.step(eaction)



class ReacherRewardShape(gym.Wrapper):

    def __init__(self, env, policy=None, shaper=None):
        super().__init__(env)
        self.policy = policy
        self.gamma = 0.99
        self.shaper = shaper
        self.prev_obs = None
        self.act0 = torch.zeros(env.action_space.shape).type(torch.float32).view(1,-1)


    def set_shaper(self, shaper):
        self.shaper = shaper

    def reset(self, **kwargs):
        obs,info = super().reset(**kwargs)
        self.prev_obs = self.observation(obs) # apply observation wrapper
        return obs,info

    def step(self, action):
        next_obs, reward, terminated, truncated, info = self.env.step(action)
        reward = self.reward(self.prev_obs,next_obs,action,reward,info)
        self.prev_obs = next_obs
        return next_obs, reward, terminated, truncated, info

    def reward(self, prev_obs, next_obs, action, reward, info):
        if self.shaper != None:
            o0 = torch.tensor(prev_obs).type(torch.float32).view(1,-1)
            o1 = torch.tensor(next_obs).type(torch.float32).view(1,-1)

            v0 = self.shaper.forward(o0, self.act0)
            v1 = self.shaper.forward(o1, self.act0)

            v0 = float(v0[0])  # take values from 1st net (???)
            v1 = float(v1[0])

            #rs = reward + (self.gamma * v1 - v0)
            #rs = info['reward_goal'] + (self.gamma * v1 - v0)
            rs = reward + (v1 - v0)
            #rs = info['reward_goal'] + (v1 - v0)
        else:
            print("Warning: shaper not provided!!!")
            rs = reward
        return rs




from gymnasium.envs.registration import register

def reachermod2(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod2(env)
    return env


def reachermod4(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod4(env)
    return env

def reachermod4x2(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod4(env)
    env = FrameStack(env,2)
    return env

def reachermod4x2sr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod4(env)
    env = FrameStack(env,2)
    env = ReacherModSR(env)
    return env

def reachermod6v(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod6v(env)
    return env

def reachermod6vsr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    return env

def reachermod6vsrrs(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    env = ReacherRewardShape(env)
    return env

def reachermod6vsrft(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherModFT(env)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    return env

def reachermod6vsra1(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherModFT(env, mode='A1')
    env = ReacherModA1(env)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    return env

def reachermod6vsrfta1(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherModFT(env, target=[-0.1*np.sqrt(2), 0.1*np.sqrt(2)])
    env = ReacherModA1(env)
    env = ReacherMod6v(env)
    env = ReacherModSR(env)
    return env

def reachermod6dsr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod6d(env)
    env = ReacherModSR(env)
    return env

def reachermod4dsr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod4d(env)
    env = ReacherModSR(env)
    return env

def reachermod4asr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod4a(env)
    env = ReacherModSR(env)
    return env

def reachermod6asr(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod6a(env)
    env = ReacherModSR(env)
    return env

def reachermod8(**args):
    env = gym.make("Reacher-v4b", **args)
    env = ReacherMod8(env)
    return env




register(
     id="Reacher-v4b",
     entry_point="rgym.envs.reacher_v4:ReacherEnv",
     max_episode_steps=50,
)




register(
     id="ReacherMod2",
     entry_point="rgym.envs.reachermod:reachermod2",
)

register(
     id="ReacherMod4",
     entry_point="rgym.envs.reachermod:reachermod4",
)

register(
     id="ReacherMod4x2",
     entry_point="rgym.envs.reachermod:reachermod4x2",
)

register(
     id="ReacherMod4x2SR",
     entry_point="rgym.envs.reachermod:reachermod4x2sr",
)

register(
     id="ReacherMod6v",
     entry_point="rgym.envs.reachermod:reachermod6v",
)

register(
     id="ReacherMod6vSR",
     entry_point="rgym.envs.reachermod:reachermod6vsr",
)

register(
     id="ReacherMod6vSRRS",
     entry_point="rgym.envs.reachermod:reachermod6vsrrs",
)

register(
     id="ReacherMod6vSRFT",
     entry_point="rgym.envs.reachermod:reachermod6vsrft",
)

register(
     id="ReacherMod6vSRA1",
     entry_point="rgym.envs.reachermod:reachermod6vsra1",
)

register(
     id="ReacherMod6vSRFTA1",
     entry_point="rgym.envs.reachermod:reachermod6vsrfta1",
)

register(
     id="ReacherMod6dSR",
     entry_point="rgym.envs.reachermod:reachermod6dsr",
)

register(
     id="ReacherMod6aSR",
     entry_point="rgym.envs.reachermod:reachermod6asr",
)

register(
     id="ReacherMod4aSR",
     entry_point="rgym.envs.reachermod:reachermod4asr",
)

register(
     id="ReacherMod4dSR",
     entry_point="rgym.envs.reachermod:reachermod4dsr",
)

register(
     id="ReacherMod8",
     entry_point="rgym.envs.reachermod:reachermod8",
)
