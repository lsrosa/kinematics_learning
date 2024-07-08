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

class ReacherFKine(gym.Wrapper):
    def __init__(self, env: gym.Env[ObsType, ActType], fkine_net, device):
        """Constructor for the observation wrapper."""
        gym.Wrapper.__init__(self, env)
        
        self.observation_space = gym.spaces.Dict({
            'x': Box(low=-np.inf, high=np.inf, shape=(2,2), dtype=np.float64)
            })
        
        self.device = device
        self.fkine_net = fkine_net
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
        return self.get_observation(), self.reward(observation,reward,info), terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self.get_observation(), info

    def get_observation(self) -> WrapperObsType:
        # joint angles
        q = np.mod(self.unwrapped.data.qpos[:2], 2*np.pi)
        qdot = self.unwrapped.data.qvel.flat[:2]           # joint ang vel
        
        _q = torche.tensor(q, dtype=torch.float32).to(self.device)
        _x_est = fkine_net(q)[:,-1]


        #xj1 = self.unwrapped.get_body_com('body1')[0:2]

        #xee = self.unwrapped.get_body_com("fingertip")[0:2]
        #_x = np.vstack((xj1, xee)).T

        obs = {'x': _x_est}
        return obs 

    def get_reward(self):
        return self.current_reward, self.current_info

    def reward(self,observation, reward, info):
        p1 = self.unwrapped.get_body_com("fingertip") - self.unwrapped.get_body_com("target")
        d1 = np.linalg.norm(p1)
        vel1 = self.unwrapped.data.qvel.flat[:2]
        v1 = np.linalg.norm(vel1)

        dn1 = np.clip(1-d1/0.4, 0, 1)           # [0,0.4] -> [1,0]
        vn1 = np.clip(1-v1/12, 0, 1)            # [0,12]  -> [1,0]

        vn1d = np.clip(math.pow(vn1,2*dn1), 0, 1)              

        g = 1.0 if (d1<0.05 and v1<1.0) else 0.0

        rv = dn1 * vn1d

        rv2 = dn1 if dn1 < 0.5 else rv

        info['reward_goal'] = g
        info['reward_vel'] = vn1d     
        info['reward_dist'] = dn1
        # info['reward_ctrl'] = np.linalg.norm(action)

        return rv + g
    
from gymnasium.envs.registration import register

def reacherfkine(**args):
    env = gym.make("Reacher-v4", **args)
    env = ReacherFKine(env)
    return env

register(
     id="ReacherFKine",
     entry_point="rgym.envs.reachertest:reachertest",
     max_episode_steps=50,
)

