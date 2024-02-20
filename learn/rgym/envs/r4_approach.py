import tcp_client

import numpy as np
from gymnasium import Env, spaces


class R4ApproachEnv(Env):
    metadata = {'render_modes': None, "render_fps": 30}

    def __init__(self):

        self.gazebo = tcp_client.Client("127.0.0.1",9000)
 
        self.start_angles = np.array([10,12,8,3])
        self.start_EEposition = np.array([0.05, 0.05, 0.17])
        self.start_die_EE = np.array([0.04,-0.02,-0.19]) 
        self.max_die_EE = np.linalg.norm(self.start_die_EE)
        

        self.object_position = np.array([1.41, 1.59, 0.01])
        
        self.angles = self.start_angles
        self.EEposition = self.start_EEposition
        self.die_EE = self.start_die_EE
        self.iteration_number = 0
        

        self.action_space = spaces.MultiDiscrete([3,3,3,3]) 
         

        self.observation_space = spaces.Dict({"angles": spaces.MultiDiscrete([25,25,25,25]),   
                                              "die_EE": spaces.Box(-np.inf, np.inf, (3,), np.float64)})


    def step(self,action):
        action = self._translate_action(action) 
        self.gazebo.msg_out = "action:%s"%str(action) 
        
        self.gazebo.send_command()

        # self.gazebo.msg_in = "OK .... Action done. angle angle angle angle x y z R P Y contact_\n\r\x00"
        state_list = self.gazebo.msg_in.split("Action done. ")[1].split("_")[0]#[:-4]

        # state_list = "angle angle angle angle x y z R P Y contact"
        state_list = state_list.split(" ")
        contact = int(state_list[-1])
        new_state = [float(v) for v in state_list if not v==""] # = [angle, angle, angle, angle, x, y, z, R, P, Y]
        
        self.angles = np.array(new_state[:4]).astype(int)
        self.die_EE = np.array(new_state[4:7]) # for the moment I do not consider RPY

        # reward computation
        reward, terminated = self._reward_computation(contact)

        observation = self._get_obs()

        self.iteration_number+=1

        return observation, reward, terminated, False, {"state": self.angles}
    
    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        self.gazebo.msg_out = "reset"
        self.gazebo.send_command()

        self.angles = self.start_angles
        self.die_EE = self.start_die_EE

        observation = self._get_obs()

        return observation, {}
    
    def _get_obs(self):
        return {"angles": self.angles, "die_EE": self.die_EE}
    
    def _translate_action(self,action):
        translated_action = [0,0,0,0]
        for i,a in enumerate(action):
            if a==0:
                translated_action[i]=-1
            elif a==1:
                translated_action[i]=0
            else: #a==2
                translated_action[i]=1
        return translated_action
    
    def _reward_computation(self,contact):
        # tolleranza delta di raggiungimento del dado: delta = 1 cm = 0.01 m
        # max_die_tip_pose = norm(die_tip_pose) quando il robot è nella sua posizione iniziale, 
        #                    oppure con un valore costante (es. 10 cm = 0.1 m)
        
        # 1 - (norm(die_tip_pose)-delta) / max_die_tip_pose
        # in questo modo il robot non tenta di raggiungere il centro del dado...
        if contact:
            return -10, False
        else:
            delta = 0.01
            distance = np.linalg.norm(self.die_EE)-delta
            terminated = distance < 0.025 # = 2.5cm
            reward = 1 - distance / self.max_die_EE
            return reward, bool(terminated)