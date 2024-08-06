import time, math, sys
import numpy as np
import torch

from stable_baselines3 import A2C, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from stable_baselines3.common.evaluation import evaluate_policy

import gymnasium as gym
from gymnasium.core import ActType, ObsType, WrapperObsType
from rgym.envs.reachertest import ReacherTest

# Import ouur model
from fkine import FKine, Jacobian, LearnCB, IncrDataset

# Utility stuff
from matplotlib import pyplot as plt
from utils import * 

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def make_env():
    env = gym.make("ReacherTest")#, render_mode = 'human') 
    return env

def play(model, env, n=5):
    i = 0
    obs,info = env.reset()
    print(obs)
    while i<n:
        action, _state = model.predict(obs)
        obs, reward, done, term, info = env.step(action)
        env.render()
        if done or term:
            obs,info = env.reset()
            i += 1
    return

def learn(seed = 2000, learn_steps=500000, random_steps=True, n_envs=32):
    log_dir = "models"
    fkine_file = log_dir + "/fkine.pt"
    
    if os.path.exists(fkine_file):
        print('models exist, no training')
        return 

    envs = make_vec_env("ReacherTest", n_envs=n_envs) 
    print("----------------------------")
    print(f"Obs: {envs.observation_space}   Act: {envs.action_space}")
    
    # train
    fkine_net = FKine(2, 2, [32, 32, 32, 32], device=device)
    fkine_net.to(device)
   
    model_cb = LearnCB()
    model_cb.env = envs 
    model_cb.fkine_net = fkine_net
    model_cb.data = IncrDataset() 
    current_seed = seed
    
    mean_losses = []

    a = np.zeros((n_envs, envs.action_space.shape[0]))
    istep = 0
    run = True
    while istep < learn_steps and run:
        print(f"{istep:6d} | Training fkine_net with seed {current_seed} ...")
        envs.seed(seed=current_seed)
        current_seed += n_envs
        obs = envs.reset()
        
        model_cb._on_rollout_start()
        for _ in range(100):
            for ienv in range(envs.num_envs):
                a[ienv] = envs.envs[ienv].action_space.sample()
            obs, _, _, _ = envs.step(a)
            istep += n_envs
            run = model_cb._on_step()
        losses = model_cb._on_rollout_end()
        mean_losses.append(losses)
   
    losses = np.array(mean_losses) 
    
    torch.save(fkine_net.state_dict(), fkine_file)

    plt.figure()
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('plots/loss.png')

    return

def test():
    log_dir = "models"
    fkine_file = log_dir + "/fkine.pt"

    env = make_env()
    y, y_dot, q, q_dot = [], [], [], []
    obs = env.reset()[0]
    
    fkine_net = FKine(2, 2, [32, 32, 32, 32], device=device)
    fkine_net.load_state_dict(torch.load(fkine_file))
    fkine_net.to(device)
    
    for epoch in range(2):
        for sample in range(50):
            action = 2*torch.rand(2)-1
            obs, reward, terminated, truncated, info = env.step(action)
            q.append(obs['q'])
            q_dot.append(obs['qdot'])
            y.append(obs['x'])
            y_dot.append(obs['xdot'])
    
    n = len(y)
    y = np.array(y)
    q = torch.Tensor(np.array(q)).to(device)
    q_dot = torch.Tensor(np.array(q_dot)).to(device)
    
    q.requires_grad_(True)
    y_pred = fkine_net(q)
    
    # get J(q)*qdot
    y_dot_pred = torch.zeros(n, 2)
    for i in range(n):
        jacobian = torch.autograd.functional.jacobian(fkine_net, q[i])
        _j = jacobian[:,:,1].reshape(2,2)
        y_dot_pred[i] = torch.matmul(_j, q_dot[i,:,None]).flatten()

    y_plot = np.zeros((n, 4))
    y_pred_plot = np.zeros((n, 4))
    
    y_dot_plot = np.zeros((n, 2))
    y_dot_pred_plot = np.zeros((n, 2))
    
    for i, label in enumerate(y):
        y_plot[i] = label.flatten()
                                                       
    for i, pred in enumerate(y_pred):
        temp = pred.cpu().detach().numpy()
        y_pred_plot[i] = temp.flatten()
    
    for i, label in enumerate(y_dot):
        y_dot_plot[i] = label[:,1].flatten()
                                                       
    for i, pred in enumerate(y_dot_pred):
        temp = pred.cpu().detach().numpy()
        y_dot_pred_plot[i] = temp.flatten()
                                                           
    plt.figure()
    plt.gca().set_prop_cycle(None)
    plt.plot(y_plot, alpha = 0.5, marker='.')
    plt.gca().set_prop_cycle(None)
    plt.plot(y_pred_plot, alpha = 0.5, marker='x')
    plt.savefig('plots/kine_predictions_x.png')
    
    plt.figure()
    plt.gca().set_prop_cycle(None)
    plt.plot(y_dot_plot, alpha = 0.5, marker='.')
    plt.gca().set_prop_cycle(None)
    plt.plot(y_dot_pred_plot, alpha = 0.5, marker='x')
    plt.savefig('plots/kine_predictions_xdot.png')
                                                       
    plt.show()

    return

if __name__ == '__main__':
    make_dirs(['models', 'plots'])
    learn()    
    test()



