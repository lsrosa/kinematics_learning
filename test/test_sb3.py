import time, math, sys
import numpy as np
import torch

from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium.envs.mujoco import ReacherEnv
from gymnasium.core import ActType, ObsType, WrapperObsType


# check device
if torch.cuda.is_available():
    print("Using GPU %s" %torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU")



'''
Reacher - https://gymnasium.farama.org/environments/mujoco/reacher/

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


def make_env():
    env = gym.make("Reacher-v4") 
    return env

def play(model, env, n=5):
    i = 0
    obs,info = env.reset()
    while i<n:
        action, _state = model.predict(obs)
        obs, reward, done, term, info = env.step(action)
        env.render()
        if done or term:
            obs,info = env.reset()
            i += 1

def learn():

    log_dir = "runs"
    log_name = "reacher_ppo"
    model_file = log_dir + "/" + log_name + ".pth"

    vec_env = make_vec_env(make_env, n_envs=32) 
    print("----------------------------")
    print(f"Obs: {vec_env.observation_space}   Act: {vec_env.action_space}")

    obs = vec_env.reset()
    print(f"Observation shape: {obs.shape}")

    # train
    try:
        model = PPO.load(model_file, vec_env)
        print(f"Model loaded from file {model_file}")
        reset_num_timesteps=False
    except Exception as e:
        print(e)
        model = PPO("MlpPolicy", vec_env,
            learning_rate = 0.01,
            tensorboard_log=log_dir,
            verbose=1)
        reset_num_timesteps=True

    print("----------------------------")
    print(model.policy)
    train_nparams = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Trainable parameters {train_nparams}")
    print("----------------------------")

    try:
        model.learn(total_timesteps=1e6, 
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=log_name,
            )
    except KeyboardInterrupt:
        print("User quit!")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model.policy, vec_env, n_eval_episodes=30)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    vec_env.close()


    # save and load
    model.save(model_file)
    print(f"Model saved on file {model_file}")

    # test
    env = gym.make("Reacher-v4", render_mode="human") 
    play(model,env)
    env.close()




if __name__ == '__main__':

    learn()    



