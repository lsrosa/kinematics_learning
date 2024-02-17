import time, math, sys, os
import argparse
import functools

import numpy as np
import torch

from stable_baselines3 import PPO, SAC, A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy


import gymnasium as gym

import rgym



# check device
if torch.cuda.is_available():
    print("Using GPU %s" %torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU")

# log/runs directory
log_dir = 'runs'

def get_names(alg, env, seed):
    global log_dir
    if type(env)==DummyVecEnv:
        env_name = env.envs[0].spec.id
    else:
        env_name = env.spec.id
    log_name = env_name + "_" + alg    
    if seed != None:
        log_name += f"_{seed}"
    model_file = log_dir + "/" + log_name + ".pth"
    lock_file = log_dir + "/" + log_name + ".lock"
    return log_dir, log_name, model_file, lock_file


def create_model(alg, env, seed=None):

    log_dir, _, model_file, _ = get_names(alg,env,seed)

    print(f"Creating model {model_file} ...")

    model = None
    if alg=="PPO":
        model = PPO("MlpPolicy", env, seed=seed,
            learning_rate = 0.005, 
            tensorboard_log=log_dir,
            verbose=0)
    elif alg=="A2C":
        model = A2C("MlpPolicy", env, seed=seed,
            learning_rate = 0.0005, 
            tensorboard_log=log_dir,
            verbose=0)
    elif alg=="SAC":
        model = SAC("MlpPolicy", env,  gradient_steps=2, seed=seed,
            verbose=0, tensorboard_log=log_dir) 
    else:
        assert False, f"Unknown algorithm {alg}"

    return model


def load_model(alg, env, seed=None):

    _, _, model_file, _ = get_names(alg,env,seed)

    print(f"Loading model {model_file} ...")

    model = None
    if alg=="PPO":
        model = PPO.load(model_file, env)
    elif alg=="A2C":
        model = A2C.load(model_file, env)
    elif alg=="SAC":
        model = SAC.load(model_file, env)
        rb_file = model_file[0:-4]+"_rb.pth"
        model.load_replay_buffer(rb_file)
    else:
        assert False, f"Unknown algorithm {alg}"

    print(f"Model loaded from file {model_file}")

    return model




def play(env_name, alg, seed, n=5):

    env = gym.make(env_name, render_mode="human") 

    model = load_model(alg, env, seed)

    for i in range(n):
        obs,info = env.reset()
        cr = 0
        cg = 1
        done = False
        while not done:
            action, _state = model.predict(obs)
            obs, reward, term, trunc, info = env.step(action)
            cr += cg * reward
            cg *= model.gamma
            env.render()
            done = term or trunc
            if done:
                print(obs)
                print(info)
                p = env.unwrapped.get_body_com("fingertip") - env.unwrapped.get_body_com("target")
                d = np.linalg.norm(p)
                v = env.unwrapped.data.qvel.flat[:2]
                v2 = np.linalg.norm(v)
                print(f"d {d:.3f} v {v2:.3f} - episode reward {cr:.3f}")
    
    env.close()


def eval_policy(env_name, alg, seed, n=100):

    env = gym.make(env_name)
    model = load_model(alg, env, seed)

    env = Monitor(env)

    print("Evaluating policy...")

    print(f"  Learning timesteps = {model.num_timesteps}")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n)
    print(f"  mean_reward = {mean_reward:.2f} +/- {std_reward}")

    env.close()



def make_env(env_name):
    env = gym.make(env_name)
    return env

def learn(env_name, alg, learn_steps=1e6, seed=None):

    vec_env = make_vec_env(functools.partial(make_env,env_name=env_name), n_envs=32) 
    print("----------------------------")
    print(f"Obs: {vec_env.observation_space}   Act: {vec_env.action_space}")

    obs = vec_env.reset()
    print(f"Observation shape: {obs.shape}")

    log_dir, log_name, model_file, lock_file = get_names(alg, vec_env, seed)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if os.path.isfile(lock_file):
        print(f"Lock found {lock_file}")
        sys.exit(1)

    
    lockf = open(lock_file, 'w')
    lockf.close()

    try:
        model = load_model(alg, vec_env, seed)
        reset_num_timesteps=False
    except Exception as e:
        print(e)
        model = create_model(alg, vec_env, seed)
        reset_num_timesteps=True

    print("----------------------------")
    print(model.policy)
    train_nparams = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Trainable parameters {train_nparams}")
    print("----------------------------")

    print(f"Learning timesteps = {model.num_timesteps}")

    try:
        print(f"Training model {model_file} ...")

        model.learn(total_timesteps=learn_steps, 
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=log_name,
            )
    except KeyboardInterrupt:
        print("User quit!")

    print(f"Learning timesteps = {model.num_timesteps}")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model.policy, vec_env, n_eval_episodes=30)
    print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

    # save and load
    model.save(model_file)
    if type(model)==SAC:
        rb_file = model_file[0:-4]+"_rb.pth"
        model.save_replay_buffer(rb_file)
    print(f"Model saved on file {model_file}")
    vec_env.close()

    os.remove(lock_file)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("-alg", type=str, default="PPO",
        help="RL algorithm")
    parser.add_argument("-env", type=str, default="Reacher-v4", 
        help="Gym environment (default: Reacher-v4)")
    parser.add_argument("-learn_steps", type=int, default=1e6,
        help="Learning steps (default: 1,000,000)")
    parser.add_argument("-seed", type=int, default=None,
        help="Random seed (default: None)")
    parser.add_argument('--play', default = False, action ='store_true', 
        help='Play one episode from saved model')
    parser.add_argument('--eval', default = False, action ='store_true', 
        help='Eval saved model')

    args = parser.parse_args()
    print(args)

    if args.play:
        play(args.env, args.alg, args.seed)
    elif args.eval:
        eval_policy(args.env, args.alg, args.seed)
    else:
        learn(args.env, args.alg, args.learn_steps, args.seed)


