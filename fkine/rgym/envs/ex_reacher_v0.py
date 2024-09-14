"""
authors: Leandro de Souza Rosa <desouzarosa@uniroma1.it>
"""

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
from pathlib import Path as path

from dm_control import mjcf

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}

def rand_between(limits):
    n = len(limits)
    ret = np.zeros(n)

    for i in range(n):
        low = limits[i,0]
        high = limits[i,1]

        ret[i] = np.random.rand()*(high-low)+low
    return ret

class ExReacherEnv(MujocoEnv, utils.EzPickle):
    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, **kwargs):
        if 'model_file' in kwargs:
            model_file = path(kwargs.pop('model_file')) 
        else:
            model_file = path.cwd()/("rgym/envs/assets/default.xml")
        #print("loading custom reacher model: %s"%model_file)
        
        # read the xml to get n_dims and n_joints
        model_xml = mjcf.from_path(model_file, escape_separators=True)
        self.n_dims = int(model_xml.find('numeric', 'ndims').data[0])
        self.n_joints = int(model_xml.find('numeric', 'njoints').data[0]) 
        self.ee_link = self.n_joints-1
        
        model_file = model_file.as_posix()
        
        self.qdot_max = 10
        
        # used to normalise the observations for the RL learning algo
        self.link_sizes = np.zeros((self.n_joints, 3))
        self.link_radii = np.zeros(self.n_joints)
        self.base_pos = model_xml.find('body', 'base').pos
        prev_radius = 0
        for j in range(self.n_joints):
            link = model_xml.find('geom', 'link%d'%j)
            self.link_sizes[j] = link.fromto[-3:] - link.fromto[:3] 
            self.link_radii[j] = prev_radius+np.linalg.norm(self.link_sizes[j])
            prev_radius = self.link_radii[j]
        
        # load model
        utils.EzPickle.__init__(self, **kwargs)
        
        obs_size = 3*self.n_joints+2*self.n_dims
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model_file,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )
        # also save joint limits
        self.joints_limits = (self.model.jnt_range)[:self.n_joints]
        
        return
    
    # get a random position in joint space
    def sample_joints(self):
        return rand_between(self.joints_limits)

    def step(self, a):
        vec = self.get_body_com("j%d"%self.ee_link)[:self.n_dims] - self.get_body_com("target")[:self.n_dims]#extended
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # velocity limit
        #extended
        vel = self.unwrapped.data.qvel.flat[:self.n_joints]
        for j in range(self.n_joints):
            if vel[j]>self.qdot_max and a[j]>0:
                a[j] = 0
            elif vel[j]<-self.qdot_max and a[j]<0:
                a[j] = 0
        
        # The output of a policy might be outside [-1, 1], so we need a clip here
        a = np.clip(a, -1, 1)

        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        # This truncated gets set to true if env has more steps then the max_episode_steps 
        ret = ob, reward, False, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        return ret 
    
    def set_joint_state(self, q):
        _qpos = np.hstack((self.sample_joints(),np.zeros(3)))
        _qvel = np.zeros(self.model.nq)
        self.set_state(_qpos, _qvel)
        return

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:self.n_joints] += self.np_random.uniform(low=-0.1, high=0.1, size=self.n_joints)

        # set random joints
        self.set_joint_state(self.sample_joints())
        
        # get goal as the fingertip position of that random position
        goal = self.get_body_com("j%d"%self.ee_link)
        self.goal = goal[:self.n_dims] 
        qpos[self.n_joints:] = goal#extended
        
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        # goal's vel is always 0
        qvel[self.n_joints:] = 0 #extended
        self.set_state(qpos, qvel)
        return self._get_obs()

    def get_observation(self):
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:self.n_joints]#extended
        obs = np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),

                # target's cart position
                self.data.qpos.flat[self.n_joints:self.n_joints+self.n_dims],#extended

                self.data.qvel.flat[:self.n_joints],#extended
                # vec between ee and target
                (self.get_body_com("j%d"%self.ee_link) - self.get_body_com("target"))[:self.n_dims],#extended
            ]
        )
        return obs 

from gymnasium.envs.registration import register

def exreacherenv(**args):
    env = ExReacherEnv(**args)
    return env

register(
     id="Ex-Reacher-v0",
     entry_point="rgym.envs.ex_reacher_v0:exreacherenv",
     max_episode_steps=50,
)
