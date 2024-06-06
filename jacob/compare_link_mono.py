import time, math, sys
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env

# Import ouur model
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from fkine.learn import learn

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

def test(models_dir, results_dir, plots_dir, model_kwargs, device):
    model_kwargs['model'] = None
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_link_file = "/fkine_FKineLinked"+_suffix
    fkine_mono_file = "/fkine_FKineMono"+_suffix
   
    n_dims  = model_kwargs['n_dims']  
    n_joints = model_kwargs['n_joints']
    env_kwargs={'n_dims':n_dims, 'n_joints':n_joints}
    env = make_env(**env_kwargs)
    y, y_dot, q, q_dot = [], [], [], []
    obs = env.reset()[0]
    
    fkine_link_net = FKineLinked(**model_kwargs, device=device)
    fkine_link_net.load_state_dict(torch.load(models_dir+fkine_link_file+".pt"))
    fkine_link_net.to(device)
    
    fkine_mono_net = FKineMono(**model_kwargs, device = device)
    fkine_mono_net.load_state_dict(torch.load(models_dir+fkine_mono_file+".pt"))
    fkine_mono_net.to(device)
    
    for sample in range(100):
        action = env.action_space.sample() 
        obs, reward, terminated, truncated, info = env.step(action)
        q.append(obs['q'])
        q_dot.append(obs['qdot'])
        y.append(obs['x'])
        y_dot.append(obs['xdot'])

    n = len(y)
    y = np.array(y)
    q_link = torch.Tensor(np.array(q)).to(device)
    q_mono = torch.Tensor(np.array(q)).to(device)
    q_dot = torch.Tensor(np.array(q_dot)).to(device)
   
    # check the model.eval() functon
    with torch.no_grad():
        y_pred_link, _ = fkine_link_net(q_link)
        y_pred_mono = fkine_mono_net(q_mono)
    
    # get J(q)*qdot just for the ee
    #TODO: maybe get for other joints
    y_dot_pred_link = torch.zeros(n, n_dims)
    y_dot_pred_mono = torch.zeros(n, n_dims)
    
    for i in range(n):
        jacobian_link,_ = torch.autograd.functional.jacobian(fkine_link_net, q_link[i])
        #print('\nlink\n', jacobian_link)
        jacobian_mono = torch.autograd.functional.jacobian(fkine_mono_net, q_mono[i])
        #print('\n\nmono\n', jacobian_mono)
        _j_link = jacobian_link[:,:,1].reshape(n_dims, n_joints)
        _j_mono = jacobian_mono[:,:,1].reshape(n_dims, n_joints)
        
        y_dot_pred_link[i] = torch.matmul(_j_link, q_dot[i,:,None]).flatten()
        y_dot_pred_mono[i] = torch.matmul(_j_mono, q_dot[i,:,None]).flatten()

    y_plot = np.zeros((n, n_dims*n_joints))
    y_pred_plot_link = np.zeros((n, n_dims*n_joints))
    y_pred_plot_mono = np.zeros((n, n_dims*n_joints))
    
    y_dot_plot = np.zeros((n, n_dims))
    y_dot_pred_plot_link = np.zeros((n, n_dims))
    y_dot_pred_plot_mono = np.zeros((n, n_dims))
    
    for i, label in enumerate(y):
        y_plot[i] = label.flatten()
                                    
    for i, pred in enumerate(y_pred_link):
        temp = pred.cpu().detach().numpy()
        y_pred_plot_link[i] = temp.flatten()
    for i, pred in enumerate(y_pred_mono):
        temp = pred.cpu().detach().numpy()
        y_pred_plot_mono[i] = temp.flatten()

    for i, label in enumerate(y_dot):
        y_dot_plot[i] = label[:,1].flatten()
                                                       
    for i, pred in enumerate(y_dot_pred_link):
        temp = pred.cpu().detach().numpy()
        y_dot_pred_plot_link[i] = temp.flatten()
    for i, pred in enumerate(y_dot_pred_mono):
        temp = pred.cpu().detach().numpy()
        y_dot_pred_plot_mono[i] = temp.flatten()

    _titles = []
    for d in range(n_dims):
        for j in range(n_joints):
            _titles += ['x%d%d'%(j,d)]
    _labels = ['ground truthl', 'linked', 'monolithic']

    plt.figure()
    for d in range(n_dims):       
        for j in range(n_joints):
            idx = j+d*n_joints
            plt.subplot(n_dims, n_joints, idx+1)
            plt.gca().set_prop_cycle(None)
            plt.plot(y_plot[:,idx], alpha = 0.5, marker='.')
            plt.plot(y_pred_plot_link[:,idx], alpha = 0.5, ls='--', marker='x')
            plt.plot(y_pred_plot_mono[:,idx], alpha = 0.5, ls = '-.', marker='s')
            plt.title(_titles[idx])
    plt.legend(_labels)
    plt.savefig(plots_dir+'/fine_predictions_x_%s.png'%_suffix, dpi=1200)
    
    _titles = []
    for d in range(n_dims):
        _titles += ['vee%d'%d]
    print(_titles)
    _titles = np.array(_titles).reshape((n_dims, 1))
    fig = plt.figure()
    for d in range(n_dims):       
        plt.subplot(n_dims, 1, d+1)
        plt.gca().set_prop_cycle(None)
        plt.plot(y_dot_plot[:,d], alpha = 0.5, marker='.')
        plt.plot(y_dot_pred_plot_link[:,d], alpha = 0.5, ls='--', marker='x')
        plt.plot(y_dot_pred_plot_mono[:,d], alpha = 0.5, ls='-.', marker='s')
        plt.title(_titles[d])
    plt.legend(_labels)
    plt.savefig(plots_dir+'/fkine_predictions_xdot_%s.png'%_suffix, dpi=1200)
                                                       
    ## compare losses
    with open(results_dir+fkine_file+'.pickle', 'rb') as h:
        losses_link, duration = pickle.load(results_dir+fkine_link_file+".pickle")
    with open(results_dir+fkine_file+'.pickle', 'rb') as h:
        losses_mono, duration = pickle.load(results_dir+fkine_mono_file+".pickle")
    plt.figure()
    plt.plot(losses_link, alpha = 0.5, ls='--', marker='x', label='linked')
    plt.plot(losses_mono, alpha = 0.5, ls='-.', marker='s', label='monolithic')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(plots_dir+'/fkine_link_x_mono_%s.png'%_suffix, dpi=1200)
    plt.show()
    return

if __name__ == '__main__':
    model_kwargs = dict()
    model_kwargs['lr'] = 1e-6#[1e-4, 1e-5, 1e-6, 1e-7]
    model_kwargs['n_hidden'] = 3#[2, 3, 4, 5]
    model_kwargs['size_hidden'] = 32#[8, 16, 32, 64]
    model_kwargs['n_joints'] = 3#[2,3]
    model_kwargs['n_dims'] = 2#[2,3]
    
    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 1000
    learn_kwargs['learn_steps'] = 100  
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['batch_size'] = 100 
    learn_kwargs['n_iter'] = 25

    model_kwargs['model'] = 'FKineLinked'
    learn('compare/models', 'compare/results', 'compare/plots', model_kwargs, learn_kwargs, device=device)
    
    model_kwargs['model'] = 'FKineMono'
    learn('compare/models', 'compare/results', 'compare/plots', model_kwargs, learn_kwargs, device=device)
    test('compare/models', 'compare/results', 'compare/plots', model_kwargs, device=device)



