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
import pickle
from glob import glob

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
   
    env_kwargs={'model_file':path.cwd()/('rgym/envs/assets/reacher%dd%dj.xml'%(model_kwargs['n_dims'], model_kwargs['n_joints']))}
    env = make_env(**env_kwargs)
    n_dims = env.unwrapped.n_dims
    n_joints = env.unwrapped.n_joints
    obs = env.reset()[0]
    
    fkine_link_model_files = sorted(glob(models_dir+fkine_link_file+"*.pt"))
    fkine_mono_model_files = sorted(glob(models_dir+fkine_mono_file+"*.pt"))
    
    error_y_link = []
    error_y_mono = []
    error_y_dot_link = []
    error_y_dot_mono = []
    
    n_samples = 100
    for fkine_link_model_file, fkine_mono_model_file in zip(fkine_link_model_files, fkine_mono_model_files):
        y, y_dot, q, q_dot = [], [], [], []
        _run_link = str.replace(fkine_link_model_file, models_dir+fkine_link_file, '').replace('.pt', '')
        _run_mono = str.replace(fkine_mono_model_file, models_dir+fkine_mono_file, '').replace('.pt', '')
        assert(_run_link == _run_mono)
        _run = _run_link

        fkine_link_net = FKineLinked(**model_kwargs, device=device)
        fkine_link_net.load_state_dict(torch.load(fkine_link_model_file))
        fkine_link_net.to(device)
        
        fkine_mono_net = FKineMono(**model_kwargs, device = device)
        fkine_mono_net.load_state_dict(torch.load(fkine_mono_model_file))
        fkine_mono_net.to(device)
        env.reset() 
        for sample in range(n_samples):
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
        
        error_y_link.append(abs(y_pred_plot_link - y_plot))
        error_y_mono.append(abs(y_pred_plot_mono - y_plot))
        error_y_dot_link.append(abs(y_dot_pred_plot_link - y_dot_plot))
        error_y_dot_mono.append(abs(y_dot_pred_plot_mono - y_dot_plot))
        
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
        plt.savefig(plots_dir+'/fine_predictions_x%s%s.png'%(_suffix, _run), dpi=1200)
        
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
        plt.savefig(plots_dir+'/fkine_predictions_xdot%s%s.png'%(_suffix, _run), dpi=1200)
    
    error_y_link = np.array(error_y_link)
    error_y_mono = np.array(error_y_mono)
    error_y_dot_link = np.array(error_y_dot_link)
    error_y_dot_mono = np.array(error_y_dot_mono)

    _titles = []
    for d in range(n_dims):
        for j in range(n_joints):
            _titles += ['x%d%d'%(j,d)]
    steps = np.linspace(1, n_samples, n_samples)
    
    plt.figure()
    for d in range(n_dims):       
        for j in range(n_joints):
            idx = j+d*n_joints
            plt.subplot(n_dims, n_joints, idx+1)
            plt.gca().set_prop_cycle(None)
            plt.plot(steps, error_y_link[:, :, idx].mean(axis=0), label='linked')
            plt.fill_between(steps, error_y_link[:,:, idx].min(axis=0), error_y_link[:, :, idx].max(axis=0), alpha=0.3)
            plt.plot(steps, error_y_mono[:, :, idx].mean(axis=0), label='monolithic')
            plt.fill_between(steps, error_y_mono[:, :, idx].min(axis=0), error_y_mono[:, :, idx].max(axis=0), alpha=0.3)
            plt.title(_titles[idx])
            plt.xlabel('step')
            plt.ylabel('absolute x error')
    plt.legend()
    plt.savefig(plots_dir+'/fine_errors_x%s.png'%(_suffix), dpi=1200)
    
    plt.figure()
    for d in range(n_dims):       
        idx = d#j+d*n_joints
        plt.subplot(n_dims, 1, idx+1)
        plt.gca().set_prop_cycle(None)
        plt.plot(steps, error_y_dot_link[:, :, idx].mean(axis=0), label='linked')
        plt.fill_between(steps, error_y_dot_link[:, :, idx].min(axis=0), error_y_dot_link[:, :, idx].max(axis=0), alpha=0.3)
        plt.plot(steps, error_y_dot_mono[:, :, idx].mean(axis=0), label='monolithic')
        plt.fill_between(steps, error_y_dot_mono[:, :, idx].min(axis=0), error_y_dot_mono[:, :, idx].max(axis=0), alpha=0.3)
        plt.title(_titles[idx])
        plt.xlabel('step')             
        plt.ylabel('absolute x dot error')
    plt.legend()
    plt.savefig(plots_dir+'/fine_errors_x_dot%s.png'%(_suffix), dpi=1200)

    with open(results_dir+fkine_link_file+'.pickle', 'rb') as h:
        losses_link, duration = pickle.load(h)
    with open(results_dir+fkine_mono_file+'.pickle', 'rb') as h:
        losses_mono, duration = pickle.load(h)
    plt.figure()
    epochs = np.linspace(1, losses_link.shape[1], losses_link.shape[1])
    
    plt.fill_between(epochs, np.min(losses_link, axis=0), np.max(losses_link, axis=0), alpha=0.3)
    plt.plot(epochs, np.mean(losses_link, axis=0), label='linked')
    plt.fill_between(epochs, np.min(losses_mono, axis=0), np.max(losses_mono, axis=0), alpha=0.3)
    plt.plot(epochs, np.mean(losses_mono, axis=0), label='monolithic')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(plots_dir+'/fkine_link_x_mono_%s.png'%_suffix, dpi=1200)
    plt.show()
    return

if __name__ == '__main__':
    model_kwargs = dict()
    model_kwargs['lr'] = 5e-4#[1e-5, 1e-6, 1e-7]
    model_kwargs['n_hidden'] = 3#[2, 3, 4, 5]
    model_kwargs['size_hidden'] = 32#[8, 16, 32, 64]
    model_kwargs['n_joints'] = 2#[2,3]
    model_kwargs['n_dims'] = 2#[2,3]
    
    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 100
    learn_kwargs['learn_steps'] = 500 
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['batch_size'] = 20#100 
    learn_kwargs['n_iter'] = 50
    learn_kwargs['append'] = False

    model_kwargs['model'] = 'FKineLinked'
    learn('compare/models', 'compare/results', 'compare/plots', model_kwargs, learn_kwargs, device=device)
    
    model_kwargs['model'] = 'FKineMono'
    learn('compare/models', 'compare/results', 'compare/plots', model_kwargs, learn_kwargs, device=device)
    test('compare/models', 'compare/results', 'compare/plots', model_kwargs, device=device)



