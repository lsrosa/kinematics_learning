from typing import TYPE_CHECKING, Optional

import numpy as np
import math

import gymnasium as gym
from gymnasium import Env, spaces, utils
from gymnasium.wrappers import FrameStack

def norm_pi(a):   # [-PI, PI)
    if a>=math.pi:
        a -= 2*math.pi
    elif a<-math.pi:
        a += 2*math.pi
    return a
    
class AbsReacher(gym.Env):

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode: Optional[str] = None,
                obs_size:int = 4,
                act_size:int = 2):

        self.params = {}

        self.params['name'] = "R2Env"

        self.params['render_mode'] = render_mode
        self.params['obs_size'] = obs_size
        self.params['act_size'] = act_size
        
        self.params['R1'] = 0.1  # radius [m]
        self.params['M1'] = 0.5  # mass [Kg]
        self.params['R2'] = 0.1  # radius [m]
        self.params['M2'] = 0.5  # mass [Kg]
        self.params['max_steps'] = 50        # max number of steps per episode
        self.params['dt'] = 0.2              # delta time [s]
        self.params['vel_scale_factor'] = 0.6  # scale from [-1,1]
        self.params['acc_scale_factor'] = 0.3  # scale from [-1,1]

        # Parameters
        
        self.render_mode = self.params['render_mode']
        self.obs_size = self.params['obs_size']
        self.act_size = self.params['act_size']
        
        self.R1 = self.params['R1']
        self.M1 = self.params['M1']
        self.R2 = self.params['R2']
        self.M2 = self.params['M2']
        self.max_steps = self.params['max_steps']
        self.dt = self.params['dt']
        self.vel_scale_factor = self.params['vel_scale_factor']
        self.acc_scale_factor = self.params['acc_scale_factor']

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32)
        self.action_space = spaces.Box( low=-1, high=+1, shape=(self.act_size,), dtype=np.float32)

        # State variables

        self.j = np.zeros((2,))    # current joint angle
        self.t = np.zeros((2,))    # target angle
        self.v = np.zeros((2,))    # ang velocity
        self.w = np.zeros((2,))    # ang acceleration
        self.n_steps = 0           # number of steps in this episode
        self.action = np.zeros((2,))  # last action executed


    def get_env_params(self):
        return self.params

    def observation(self, observation=None):  # for compatibility with Observation Wrappers
        return self.get_observation()

    def reward(self, observation=None):  # for compatibility with Observation Wrappers
        return self.get_reward()
        
    def get_observation(self):
        if self.obs_size == 4:
            obs = np.array([self.j[0], self.j[1], 
                self.tpos[0], self.tpos[1] ], dtype=np.float32)
        elif self.obs_size == 6:
            obs = np.array([self.j[0], self.j[1], 
                self.tpos[0], self.tpos[1], self.v[0], self.v[1] ], dtype=np.float32)
        else:
            assert False, f"AbsReacher: Wrong observation size {self.obs_size}"

        return obs

    def get_action(self):
        return self.action

    def get_reward(self):
        dist = np.linalg.norm(np.array(self.tpos)-np.array(self.xpos))
        info = { 'dist': dist }
        v2 = np.linalg.norm(self.v)
        info['v2'] = v2
        w2 = np.linalg.norm(self.w)
        info['w2'] = w2
        reward = - dist - v2 - w2
        return reward, info

    def fk(self):
        self.xpos = [ self.R1*math.cos(self.j[0])+self.R2*math.cos(self.j[0]+self.j[1]), 
                      self.R1*math.sin(self.j[0])+self.R2*math.sin(self.j[0]+self.j[1]) ]  # robot ee pos

    def reset(self,*,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
        ):
        super().reset(seed=seed)
        self.j[0] = self.np_random.uniform(-math.pi,math.pi) # robot joint angle
        self.j[1] = self.np_random.uniform(-math.pi,math.pi) # robot joint angle
        self.t[0] = self.np_random.uniform(-math.pi,math.pi) # target angle
        self.t[1] = self.np_random.uniform(-math.pi,math.pi) # target angle
        self.fk()
        self.tpos = [ self.R1*math.cos(self.t[0])+self.R2*math.sin(self.t[0]+self.t[1]), 
                      self.R1*math.sin(self.t[0])+self.R2*math.sin(self.t[0]+self.t[1]) ]  # target pos
        self.v = np.zeros((2,))    # ang velocity
        self.w = np.zeros((2,))    # ang acceleration
        self.n_steps = 0

        reward, info = self.get_reward()

        return self.get_observation(), info


    # action = angular velocity (discrete)
    def step(self, action):

        self.action[0] = action[0]  # [-1,1]
        self.action[1] = action[1]

        self.w[0] = self.acc_scale_factor * self.action[0]  
        self.w[1] = self.acc_scale_factor * self.action[1]

        self.v[0] = self.v[0] + self.w[0] * self.dt
        self.v[1] = self.v[1] + self.w[1] * self.dt
        self.j[0] = norm_pi(self.j[0] + self.v[0] * self.dt)
        self.j[1] = norm_pi(self.j[1] + self.v[1] * self.dt)
        self.fk()  # robot ee pos 

        self.n_steps += 1

        terminated = False
        truncated = self.n_steps >= self.max_steps
        reward, info = self.get_reward()

        return self.get_observation(), reward, terminated, truncated, info


    def render(self):
        rew, info = self.get_reward()
        dist = info['dist']
        print(f"J {self.j[0]:6.3f} {self.j[1]:6.3f} | T ({self.t[0]:6.3f} {self.t[1]:6.3f}) {self.tpos[0]:6.3f} {self.tpos[1]:6.3f} | d ({dist:6.3f}) | A {self.action[0]:6.3f} {self.action[1]:6.3f} | v {self.v[0]:6.3f} {self.v[1]:6.3f} | rew: {rew:8.3f}")
        return

    def close(self):
        pass


from gymnasium.envs.registration import register

def absreacher4(**args):
    env = AbsReacher(obs_size = 4)
    return env

def absreacher4x2(**args):
    env = AbsReacher(obs_size = 4)
    env = FrameStack(env,2)
    return env

def absreacher6v(**args):
    env = AbsReacher(obs_size = 6)
    return env

register(
     id="AbsReacher4",
     entry_point="rgym.envs.absreacher:absreacher4",
     max_episode_steps=50,
)

register(
     id="AbsReacher4x2",
     entry_point="rgym.envs.absreacher:absreacher4x2",
     max_episode_steps=50,
)

register(
     id="AbsReacher6v",
     entry_point="rgym.envs.absreacher:absreacher6v",
     max_episode_steps=50,
)
