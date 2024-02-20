import tcp_client

import numpy as np
from gymnasium import Env, spaces


class R4PusherEnv(Env):
    metadata = {'render_modes': None, "render_fps": 30}

    def __init__(self, reset_arm_position, reset_die_pose):

        self.gazebo = tcp_client.Client("127.0.0.1",9000)

        self.reset_arm_position = reset_arm_position
        self.reset_die_pose = reset_die_pose

        # in this sub-task we move only the third and fourth joints
        self.start_angles = np.array([self.reset_arm_position[1],self.reset_arm_position[2],self.reset_arm_position[3]])
        self.start_die_pose = np.array(self.reset_die_pose)
        
        self.die_pose = self.start_die_pose
        self.angles = self.start_angles

        
        self.previous_die_pose = self.start_die_pose

        self.iteration_number = 0
        self.no_contact_count = 0
        self.contact_episode_number = 0

        self.action_space = spaces.MultiDiscrete([3,3,3]) 

        self.observation_space = spaces.Dict({"angles": spaces.MultiDiscrete([25,25,25]),   
                                              "die_pose": spaces.Box(-np.inf, np.inf, (6,), np.float64)})
        
    def step(self,action):
        action = self._translate_action(action)
        self.gazebo.msg_out = "action:%s"%str(action) #[0,a2,a3,a4]
    
        self.gazebo.send_command()
        
        # self.gazebo.msg_in = "OK .... Action done. angle angle angle angle x y z R P Y \n\r\x00"
        state_list = self.gazebo.msg_in.split("Action done. ")[1].split("_")[0]#[:-4]
        # state_list = "angle angle angle angle x y z R P Y contact"
        state_list = state_list.split(" ")
        contact = int(state_list[-1])
        new_state = [float(v) for v in state_list if not v==""] # = [angle, angle, angle, angle, x, y, z, R, P, Y]
        
        self.angles = np.array(new_state[1:4]).astype(int)
        self.previous_die_pose = self.die_pose
        self.die_pose = np.array(new_state[4:10]) 

        # reward computation
        reward, terminated = self._reward_computation(contact)

        observation = self._get_obs()

        self.iteration_number+=1

        return observation, reward, terminated, False, {"state": self.angles}
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        arm_positon = ""
        for v in self.reset_arm_position:
            arm_positon+=(str(v)+" ")
        die_positon = ""
        for v in self.reset_die_pose:
            die_positon+=(str(v)+" ")
        self.gazebo.msg_out = "reset:"+arm_positon+die_positon[:-1]
        self.gazebo.send_command()
        
        self.angles = self.start_angles
        self.die_pose = self.start_die_pose
        self.previous_die_pose = self.start_die_pose
        self.no_contact_count = 0
        self.contact_episode_number += 1

        observation = self._get_obs()

        return observation, {}
    
    def _get_obs(self):
        return {"angles": self.angles, "die_pose": self.die_pose}
    
    def _translate_action(self,action):
        translated_action = [0,0,0,0]
        for i,a in enumerate(action): #[a2,a3,a4]
            if a==0:
                translated_action[i+1]=-1
            elif a==1:
                translated_action[i+1]=0
            else: #a==2
                translated_action[i+1]=1
        return translated_action
    
    def _reward_computation(self,contact):
        die_position = self.die_pose[:3]
        #previous_die_position = self.previous_die_pose[:3]
        start_die_position = self.start_die_pose[:3]
        die_orientation = self.die_pose[3:]
        #previous_die_orientation = self.previous_die_pose[3:]
        start_die_orientation = self.start_die_pose[3:]


        reward_position = np.linalg.norm(die_position - start_die_position)
        reward_orientation = np.linalg.norm(die_orientation - start_die_orientation)
        reward = reward_position + reward_orientation
        #reward = np.linalg.norm(self.die_pose-self.previous_die_pose)
        if contact:
            reward = -10
            return -10, -10, reward, bool(False)
        if np.array_equal(np.round(self.die_pose,2), np.round(self.previous_die_pose,2)):
            self.no_contact_count += 1
            reward -= 0.025 * self.no_contact_count * self.contact_episode_number
        else:
            self.no_contact_count = 0
            

        if reward_position >= 0.01 or reward_orientation >= 0.2:
            self.contact_episode_number = 0
            return reward_position, reward_orientation, reward, bool(True)

        return reward, bool(False)