import time, math, sys, os
import argparse
import functools

import numpy as np
import torch

from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

#print(sys.path)
#sys.path.insert(0, '../Gymnasium' )
#print(sys.path)

import gymnasium as gym

import rgym

from networks import *


# check device
if torch.cuda.is_available():
    print("Using GPU %s" %torch.cuda.get_device_name(torch.cuda.current_device()))
else:
    print("Using CPU")

# log/runs directory
log_dir = 'runs'

def get_names(alg, env, seed, fknet=None):
    global log_dir
    if type(env)==DummyVecEnv:
        env_name = env.envs[0].spec.id
    else:
        env_name = env.spec.id
    log_name = env_name + "_" + alg
    if fknet != None:
        log_name += "_FK"
    if seed != None:
        log_name += f"_{seed}"
    model_file = log_dir + "/" + log_name + ".pth"
    lock_file = log_dir + "/" + log_name + ".lock"
    return log_dir, log_name, model_file, lock_file


def create_model(alg, env, seed=None, fknet=None):

    log_dir, _, model_file, _ = get_names(alg,env,seed,fknet)

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
        model = SAC("MlpPolicy", env, seed=seed, gradient_steps=4, 
            verbose=0, tensorboard_log=log_dir)
    elif alg=="TD3":
        model = TD3("MlpPolicy", env,  seed=seed,
            verbose=0, tensorboard_log=log_dir)
    else:
        assert False, f"Unknown algorithm {alg}"

    return model


def load_model(alg, env, seed=None, fknet=None):

    _, _, model_file, _ = get_names(alg,env,seed,fknet)

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
    elif alg=="TD3":
        model = TD3.load(model_file, env)
        rb_file = model_file[0:-4]+"_rb.pth"
        model.load_replay_buffer(rb_file)
    else:
        assert False, f"Unknown algorithm {alg}"

    print(f"Model loaded from file {model_file}")

    return model




def play(env_name, alg, seed, n=5, fknet_file=None, render=True):

    if render:
        render_mode="human"
    else:
        render_mode= None

    env = gym.make(env_name, render_mode=render_mode)

    fknet = None
    if fknet_file != None:
        fknet = FKNet(n_in=2, n_out=2)
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
        env.set_fknet(fknet)

    model = load_model(alg, env, seed, fknet)

    goals = 0
    cnt_ep = 0
    sum_epreward = 0
    sum2_epreward = 0
    for i in range(n):
        print(f"-------------- {i}/{n} -----------------")
        try:
            obs,info = env.reset()
            cr = 0
            cg = 1
            done = False
            while not done:
                action, _state = model.predict(obs)
                obs, reward, term, trunc, info = env.step(action)
                cr += cg * reward
                #cg *= model.gamma
                if render:
                    env.render()
                    time.sleep(0.05)
                done = term or trunc
                if done:
                    print(f"obs {vstr(obs)}")
                    print(info)
                    goals += info['reward_goal']
                    sum_epreward += cr
                    sum2_epreward += cr * cr
                    cnt_ep += 1
                    try:
                        # Reacher
                        p = env.unwrapped.get_body_com("fingertip") - env.unwrapped.get_body_com("target")
                        d = np.linalg.norm(p)
                        v = env.unwrapped.data.qvel.flat[:2]
                        v2 = np.linalg.norm(v)
                        print(f"d {d:.3f} v {v2:.3f} - episode reward {cr:.3f}")
                    except:
                        pass

                    try:
                        # Pusher
                        vec_1 = env.unwrapped.get_body_com("object") - env.unwrapped.get_body_com("tips_arm")
                        vec_2 = env.unwrapped.get_body_com("object") - env.unwrapped.get_body_com("goal")
                        vel = env.unwrapped.data.qvel.flat[:7]
                        d1 = np.linalg.norm(vec_1)
                        d2 = np.linalg.norm(vec_2)
                        v = np.linalg.norm(vel)

                        print(f"d1 {d1:.3f} d2 {d2:.3f} v {v:.3f} - episode reward {cr:.3f}")
                    except Exception as e:
                        #print(e)
                        pass
        except KeyboardInterrupt:
            print("User quit!")
            break

    mean_reward = sum_epreward/cnt_ep
    std_reward = math.sqrt(sum2_epreward/cnt_ep - math.pow(mean_reward,2))

    print(f"Learning timesteps = {model.num_timesteps}")

    print(f"mean_reward = {mean_reward:.2f} +/- {std_reward:.2f}")

    print(f"Success rate: {int(goals)}/{cnt_ep} = {goals/n*100:.1f}%")

    env.close()


def eval_policy(env_name, alg, seed, n=100, fknet_file=None):

    env = gym.make(env_name)

    fknet = None
    if fknet_file != None:
        fknet = FKNet(n_in=2, n_out=2)
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
        env.set_fknet(fknet)

    model = load_model(alg, env, seed, fknet)

    env = Monitor(env)

    print("Evaluating policy...")

    print(f"  Learning timesteps = {model.num_timesteps}")

    # Evaluate the policy
    mean_reward, std_reward = evaluate_policy(model.policy, env, n_eval_episodes=n)
    print(f"  mean_reward = {mean_reward:.2f} +/- {std_reward}")

    env.close()

import fknet_learn

def make_env(env_name,fknet):
    env = gym.make(env_name)
    if fknet!=None:
        env.set_fknet(fknet)
    return env

def learn(env_name, alg, learn_steps=1e6, seed=None, fknet_file=None, shaper_file=None):

    fknet = None
    if fknet_file != None:
        _, n_joints, n_out = fknet_learn.get_info(env_name)
        fknet = FKNet(n_in=n_joints, n_out=n_out)
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")

    shaper = None
    if shaper_file != None:
        shaper_model = SAC.load(shaper_file, None)
        shaper = shaper_model.policy.critic
        print(f"Loaded reward shaper {shaper_file}")

    num_envs = 32
    if alg == 'TD3':
        num_envs = 1

    vec_env = make_vec_env(functools.partial(make_env,env_name=env_name,fknet=fknet), n_envs=num_envs)
    print("----------------------------")
    print(f"Obs: {vec_env.observation_space}   Act: {vec_env.action_space}")

    obs = vec_env.reset()
    print(f"Observation shape: {obs.shape}")


    log_dir, log_name, model_file, lock_file = get_names(alg, vec_env, seed, fknet)

    if not os.path.isdir(log_dir):
        os.mkdir(log_dir)

    if os.path.isfile(lock_file):
        print(f"Lock found {lock_file}")
        sys.exit(1)


    try:
        model = load_model(alg, vec_env, seed, fknet)
        reset_num_timesteps=False
    except Exception as e:
        print(e)
        model = create_model(alg, vec_env, seed, fknet)
        reset_num_timesteps=True

    print("----------------------------")
    print(model.policy)
    train_nparams = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print(f"Trainable parameters {train_nparams}")
    print("----------------------------")

    # setting fknet and shaper

    if fknet != None:
        for env in vec_env.envs:
            env.set_fknet(fknet)

    if shaper != None:
        for env in vec_env.envs:
            env.set_shaper(shaper)

    print(f"Learning timesteps = {model.num_timesteps}")

    lockf = open(lock_file, 'w')
    lockf.close()

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
    if type(model)==SAC or type(model)==TD3:
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
    parser.add_argument("-fknet", type=str, default=None,
        help="FKNet model (default: None)")
    parser.add_argument("-shaper", type=str, default=None,
        help="Reward shaping (value net of a policy) (default: None)")
    parser.add_argument('--play', default = False, action ='store_true',
        help='Play one episode from saved model')
    parser.add_argument('--eval', default = False, action ='store_true',
        help='Eval saved model')

    args = parser.parse_args()
    print(args)

    if args.play:
        play(args.env, args.alg, args.seed, n=10, fknet_file=args.fknet, render=True)
    elif args.eval:
        play(args.env, args.alg, args.seed, n=100, fknet_file=args.fknet, render=False)
    else:
        learn(args.env, args.alg, args.learn_steps, args.seed, args.fknet, args.shaper)



