# -------------------------------------------------------------------------------- #
# Here is the training of the model so that the robotic arm gets close to the die
# -------------------------------------------------------------------------------- #


import datetime

import gymnasium as gym
from gymnasium.envs.registration import register
from gymnasium.wrappers import FlattenObservation

from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env

MAX_EPISODE_STEPS = 80

register(
     id="R4Approach-v0",
     entry_point="r4_approach:R4ApproachEnv",
     max_episode_steps=MAX_EPISODE_STEPS,
)

class TensorboardCallback(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
        }
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
            "train/learning_rate": 0.0,
            "time/iterations": 0,
            "time/total_timesteps": 0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )
    def _on_step(self) -> bool:
        return True
    

def EnvModelCreation(nameEnv,logFile):
    env = gym.make(nameEnv) 
    env = FlattenObservation(env)
    env = Monitor(env)
    model = PPO(MlpPolicy, env, 
                n_steps = 80, batch_size = 10, learning_rate = 5e-4, clip_range = 0.55,
                verbose=1, tensorboard_log=logFile)
    
    return env, model

def learningFunction(model,nameModel,timesteps):
    model.learn(total_timesteps=timesteps, progress_bar=True, callback=TensorboardCallback())
    
    model.save(nameModel)
    del model

def evaluationAgent(nameModel, nameEnv, n_episodes):
    model = PPO.load(nameModel)
    
    env = gym.make(nameEnv)
    env = FlattenObservation(env)
    env = Monitor(env)
    obs, _ = env.reset()

    tot_reward = 0

    for _ in range(MAX_EPISODE_STEPS):
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, _ = env.step(action)
        tot_reward += reward
        if terminated:
            print("The episode is terminated")
            break
    obs, _ = env.reset()

timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
nameEnv = "R4Approach-v0"
nameModel = "PPO_"+nameEnv+"_"+timestamp
logFile = "./tensorboard/"+str(timestamp)


env, model = EnvModelCreation(nameEnv,logFile)
learningFunction(model, nameModel, MAX_EPISODE_STEPS*210)#100000
evaluationAgent(nameModel, nameEnv, 1)
