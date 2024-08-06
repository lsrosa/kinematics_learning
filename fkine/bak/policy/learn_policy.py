import time, math, sys
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym

# Import our model
from jacobian import Jacobian
from rgym.envs.reacherfkine import ReacherFKine

# Utility stuff
from matplotlib import pyplot as plt
from utils import * 
from sac import Agent

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def learn(seed = 2000, learn_steps=100000, random_steps=True, n_envs=32):
    log_dir = "runs"
    log_name = "reacher_SAC_policy"
    model_file = log_dir + "/" + log_name + ".pth"

    env = gym.make("ReacherFKine") 
    print("----------------------------")
    print(f"Obs: {env.observation_space}   Act: {env.action_space}")
    nq = env.observation_space['q'].shape[0]
    # Load FK model
    fkine_net = Jacobian(nq, 2, [32, 32, 32, 32], device=device)
    fkine_net.load_state_dict(torch.load('models/jacob.pt'))
    fkine_net.to(device)
    sac_agent = Agent(input_dims=2, env=env, n_actions=2)
    
    mean_losses = []
    istep = 0
    run = True
    
    seed = 3000
    while istep < learn_steps and run:
        print(f"{istep:6d} | Training fknet with seed {seed} ...")
        seed += n_envs
        obs, _ = env.reset() # also returns info
        q = torch.tensor(obs['q'].reshape(1, nq), dtype=torch.float32).to(device)
        x_obs = fkine_net(q)[:,-1] #just last joint (fingertip)
        reward_acc = 0
        for steps in range(300):
            a = sac_agent.choose_action(x_obs)
            obs_, reward, done, _, _ = env.step(a)
            reward_acc += reward
            q_ = torch.tensor(obs_['q'].reshape(1, nq), dtype=torch.float32).to(device)
            x_obs_ = fkine_net(q_)[:,-1]
            sac_agent.remember(x_obs.cpu().detach().numpy(), a, reward, x_obs_.cpu().detach().numpy(), done)
            sac_agent.learn()
            x_obs = x_obs_
        istep += steps
        avg_reward = reward_acc/steps
        print(reward)
        mean_losses.append(avg_reward)
   
    losses = np.array(mean_losses) 
    #torch.save(net.state_dict(), 'models/jacob.pt')
    plt.figure()
    plt.plot(losses)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig('plots/loss.png')

    return


def test():
    env = make_env()
    y, y_dot, q, q_dot = [], [], [], []
    obs = env.reset()[0]
    
    net = Jacobian(2, 2, [32, 32, 32, 32], device=device)
    net.load_state_dict(torch.load('models/jacob.pt'))
    net.to(device)
    
    for epoch in range(2):
        for sample in range(50):
            action = 2*torch.rand(2)-1
            obs, reward, terminated, truncated, info = env.step(action)
            q.append(obs['q'])
            q_dot.append(obs['qdot'])
            y.append(obs['x'])
            #y_dot.append(obs['xdot'])
    
    n = len(y)
    y = np.array(y)
    q = torch.Tensor(np.array(q)).to(device)
    q_dot = torch.Tensor(np.array(q_dot)).to(device)

    with torch.no_grad():
        y_pred = net(q).cpu().detach().numpy()
        y_dot_pred = net(q_dot).cpu().detach().numpy()

    y_plot = np.zeros((n, 4))
    y_pred_plot = np.zeros((n, 4))
    
    y_dot_plot = np.zeros((n, 4))
    y_dot_pred_plot = np.zeros((n, 4))
    
    for i, label in enumerate(y):
        y_plot[i] = label.flatten()

    for i, pred in enumerate(y_pred):
        y_pred_plot[i] = pred.flatten()
   
    for i, label in enumerate(y_dot):
        y_dot_plot[i] = label.flatten()

    for i, pred in enumerate(y_dot_pred):
        y_dot_pred_plot[i] = pred.flatten()

    plt.figure()
    plt.plot(y_plot, alpha = 0.5, marker='.')
    plt.plot(y_pred_plot, alpha = 0.5, marker='x')
    plt.savefig('plots/jacob_predictions_x.png')
    
    plt.figure()
    plt.plot(y_dot_plot, alpha = 0.5, marker='.')
    plt.plot(y_dot_pred_plot, alpha = 0.5, marker='x')
    plt.savefig('plots/jacob_predictions_xdot.png')

    plt.show()

    return

if __name__ == '__main__':
    make_dirs(['models', 'plots'])
    learn()    
