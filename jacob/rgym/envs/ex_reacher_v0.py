"""
authors: Leandro de Souza Rosa <desouzarosa@uniroma1.it>
"""

import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os

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
    """
    ## Description
    "Reacher" is a *nj*-jointed robot arm. The goal is to move the robot's end effector (fingertip, called body*nj*) close to a
    target that is spawned at a random position in an *nd*  cartesian space (usually *nd* = 2 or 3).

    ## Action Space
    The action space is a `Box(-1, 1, (*nj*,), float32)`. An action represents the torques applied at the hinge joints.

    | Num | Action                                                                          | Control Min | Control Max | Name (in corresponding XML file) | Joint | Unit |
    |-----|---------------------------------------------------------------------------------|-------------|-------------|--------------------------|-------|------|
    | 0   | Torque applied at hinges (connecting the link to the point of fixture) | -1 | 1 | joint[0 to *nj*]  | hinge | torque (N m) |

    ## Observation Space
    Observations consist of

    - The cosine of the angles of the arms
    - The sine of the angles of the arms
    - The coordinates of the target
    - The angular velocities of the arms
    - The vector between the target and the reacher's fingertip (3 dimensional with the last element being 0)

    The observation is a `Box(-Inf, Inf, (11,), float64)` where the elements correspond to the following:

    | Num | Observation                                                                                    | Min  | Max | Name (in corresponding XML file) | Joint | Unit                     |
    | --- | ---------------------------------------------------------------------------------------------- | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0 to *nj*-1   | cosine of the angle of the arms                                                           | -Inf | Inf | cos(joint[0 to *nj*])                      | hinge | unitless                 |
    | *nj* to 2*nj*-1   | sine of the angle of the armd                                                             | -Inf | Inf | sin(joint[0 to *nj*])                      | hinge | unitless                 |
    | 2*nj* to 2*nj*+*nd* - 1   | cartesian coordinates of the target                                                                     | -Inf | Inf | target_x                         | slide | position (m)             |
    | 2*nj*+*nd* to 3*nj*+*nd*-1   | angular velocity of the arms                                                              | -Inf | Inf | joint[0 to n]                           | hinge | angular velocity (rad/s) |
    | 3*nj*+*nd* to 3*nj*+2*nd*-1   | cartesiain position_fingertip - position_target                                                | -Inf | Inf | NA                               | slide | position (m)             |


    ## Rewards
    The reward consists of two parts:
    - *reward_distance*: This reward is a measure of how far the *fingertip*
    of the reacher (the unattached end) is from the target, with a more negative
    value assigned for when the reacher's *fingertip* is further away from the
    target. It is calculated as the negative vector norm of (position of
    the fingertip - position of target), or *-norm("fingertip" - "target")*.
    - *reward_control*: A negative reward for penalising the walker if
    it takes actions that are too large. It is measured as the negative squared
    Euclidean norm of the action, i.e. as *- sum(action<sup>2</sup>)*.

    The total reward returned is ***reward*** *=* *reward_distance + reward_control*

    Unlike other environments, Reacher does not allow you to specify weights for the individual reward terms.
    However, `info` does contain the keys *reward_dist* and *reward_ctrl*. Thus, if you'd like to weight the terms,
    you should create a wrapper that computes the weighted reward from `info`.

    Most Gym environments just return the positions and velocity of the
    joints in the `.xml` file as the state of the environment. However, in
    reacher the state is created by combining only certain elements of the
    position and velocity, and performing some function transformations on them.
    If one is to read the `.xml` for reacher then they will find 4 joints:
                                                                                                                                
    | Num | Observation                 | Min      | Max      | Name (in corresponding XML file) | Joint | Unit               |
    |-----|-----------------------------|----------|----------|----------------------------------|-------|--------------------|
    | 0   | angle of the first arm      | -Inf     | Inf      | joint0                           | hinge | angle (rad)        |
    | 1   | angle of the second arm     | -Inf     | Inf      | joint1                           | hinge | angle (rad)        |
    | 2   | x-coordinate of the target  | -Inf     | Inf      | target_x                         | slide | position (m)       |
    | 3   | y-coordinate of the target  | -Inf     | Inf      | target_y                         | slide | position (m)       |

    ## Starting State
    All observations start in state
    (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    with a noise added for stochasticity. A uniform noise in the range
    [-0.1, 0.1] is added to the positional attributes, while the target position
    is selected uniformly at random in a disk of radius 0.2 around the origin.
    Independent, uniform noise in the
    range of [-0.005, 0.005] is added to the velocities, and the last
    element ("fingertip" - "target") is calculated at the end once everything
    is set. The default setting has a framerate of 2 and a *dt = 2 * 0.01 = 0.02*

    ## Episode End

    The episode ends when any of the following happens:

    1. Truncation: The episode duration reaches a 50 timesteps (with a new random target popping up if the reacher's fingertip reaches it before 50 timesteps)
    2. Termination: Any of the state space values is no longer finite.

    ## Arguments

    No additional arguments are currently supported (in v2 and lower),
    but modifications can be made to the XML file in the assets folder
    (or by changing the path to a modified XML file in another folder)..

    ```python
    import gymnasium as gym
    env = gym.make('Reacher-v4')
    ```

    There is no v3 for Reacher, unlike the robot environments where a v3 and
    beyond take `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc.

    ## Version History

    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3
    * v2: All continuous control environments now use mujoco-py >= 1.50
    * v1: max_time_steps raised to 1000 for robot based tasks (not including reacher, which has a max_time_steps of 50). Added reward_threshold to environments.
    * v0: Initial versions release (1.0.0)
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 50,
    }

    def __init__(self, n_dims=2, n_joints=2, **kwargs):
        self.n_dims = n_dims 
        self.n_joints = n_joints 
        
        self.ee_link = self.n_joints-1

        ext = str(self.n_dims)+'d'+str(self.n_joints)+"j"
        model = os.getcwd()+"/rgym/envs/assets/reacher"+ext+".xml"
        #print("loading custom reacher model: %s"%model)

        utils.EzPickle.__init__(self, **kwargs)
        
        obs_size = 3*self.n_joints+2*self.n_dims
        #print("observation space size: ", obs_size) 
        observation_space = Box(low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64)
        MujocoEnv.__init__(
            self,
            model,
            2,
            observation_space=observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            **kwargs,
        )

    def step(self, a):
        vec = self.get_body_com("j%d"%self.ee_link)[:self.n_dims] - self.get_body_com("target")[:self.n_dims]#extended
        reward_dist = -np.linalg.norm(vec)
        reward_ctrl = -np.square(a).sum()
        reward = reward_dist + reward_ctrl

        # velocity limit
        #extended
        vel = self.unwrapped.data.qvel.flat[:self.n_joints]
        for j in range(self.n_joints):
            if vel[j]>10 and a[j]>0:
                a[j] = 0
            elif vel[j]<-10 and a[j]<0:
                a[j] = 0
        self.do_simulation(a, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()
        ret = ob, reward, False, False, dict(reward_dist=reward_dist, reward_ctrl=reward_ctrl)
        #print('\n\nin ex: ', ret)
        return ret 

    def reset_model(self):
        qpos = self.init_qpos
        qpos[:self.n_joints] += self.np_random.uniform(low=-0.1, high=0.1, size=self.n_joints)

        #extended
        # get a random position in joint space
        joints_limits = (self.model.jnt_range)[:self.n_joints]
        _qpos = np.hstack((rand_between(joints_limits),np.zeros(3)))
        _qvel = np.zeros(self.model.nq)
        self.set_state(_qpos, _qvel)
        
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
