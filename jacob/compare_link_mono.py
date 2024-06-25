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
import seaborn, pandas, itertools
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

def test(models_dir, results_dir, plots_dir, model_kwargs_link, model_kwargs_mono, device):
    model_kwargs_link['model'] = None
    model_kwargs_mono['model'] = None
    _suffix_link = model_kwargs_2_str(**model_kwargs_link)
    _suffix_mono = model_kwargs_2_str(**model_kwargs_mono)
    fkine_link_file = "/fkine_FKineLinked"+_suffix_link
    fkine_mono_file = "/fkine_FKineMono"+_suffix_mono
    n_dims = model_kwargs_link['n_dims'] 
    n_joints = model_kwargs_link['n_joints'] 

    env_kwargs={'model_file':path.cwd()/('rgym/envs/assets/reacher%dd%dj.xml'%(n_dims, n_joints))}
    env = make_env(**env_kwargs)
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

        fkine_link_net = FKineLinked(**model_kwargs_link, device=device)
        fkine_link_net.load_state_dict(torch.load(fkine_link_model_file, map_location=torch.device(device)))
        fkine_link_net = fkine_link_net.to(device)

        fkine_mono_net = FKineMono(**model_kwargs_mono, device=device)
        fkine_mono_net.load_state_dict(torch.load(fkine_mono_model_file, map_location=torch.device(device)))
        fkine_mono_net = fkine_mono_net.to(device)
        
        env.reset() 
        for sample in range(n_samples):
            if np.random.rand()>0.8:
                env.reset()
            action = env.action_space.sample() 
            obs, reward, terminated, truncated, info = env.step(action)
            q.append(obs['q'].copy())
            q_dot.append(obs['qdot'].copy())
            y.append(obs['x'].copy())
            y_dot.append(obs['xdot'].copy())
        
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
            jacobian_mono = torch.autograd.functional.jacobian(fkine_mono_net, q_mono[i])
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
        _suffix = '%dd%dj'%(n_dims, n_joints) 
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
        plt.savefig(plots_dir+'/fkine_predictions_x%s%s.png'%(_suffix, _run), dpi=1200)
        plt.close()

        _titles = []
        for d in range(n_dims):
            _titles += ['vee%d'%d]

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
        plt.close() 

        error_joints_link = (y_pred_link.cpu() - y).norm(dim=1).transpose(0,1)
        error_joints_mono = (y_pred_mono.cpu() - y).norm(dim=1).transpose(0,1)
        _min_error = min(error_joints_link.min(), error_joints_mono.min())
        _max_error = max(error_joints_link.max(), error_joints_mono.max())
        bins = np.linspace(_min_error*0.9, _max_error*1.1, 20)
        labels = ['Linked', 'Monolithic']
        fig, axs = plt.subplots(1, n_joints, sharex='all', sharey='all', figsize=(4*n_joints, 4*1))
        palette = itertools.cycle(seaborn.color_palette())
        color = [[np.array(next(palette))] for i in range(2)]
        for j, ax in enumerate(axs.flat):
            plt.gca().set_prop_cycle(None)
            df = pandas.DataFrame({'Linked': error_joints_link[j]})
            seaborn.histplot(data=df, bins=bins, palette=color[0], stat='percent', shrink=0.8, ax=ax, legend=False, common_bins=False, kde=False)
            df = pandas.DataFrame({'Monolithic': error_joints_mono[j]})
            seaborn.histplot(data=df, bins=bins, palette=color[1], stat='percent', shrink=0.8, ax=ax, legend=False, common_bins=False, kde=False)
            ax.title.set_text('Joint %d'%(j+1))
            ax.set_xlabel('Error')
            if j == 0:
                ax.set_ylabel('# Samples (%)')
            if j == n_joints-1:
                plt.legend(labels)
        plt.savefig(plots_dir+'/fkine_errors_hist_x%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close() 

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
            plt.fill_between(steps, error_y_link[:, :, idx].min(axis=0), error_y_link[:, :, idx].max(axis=0), alpha=0.3)
            plt.plot(steps, error_y_mono[:, :, idx].mean(axis=0), label='monolithic')
            plt.fill_between(steps, error_y_mono[:, :, idx].min(axis=0), error_y_mono[:, :, idx].max(axis=0), alpha=0.3)
            plt.title(_titles[idx])
            plt.xlabel('step')
            plt.ylabel('absolute x error')
    plt.legend()
    plt.savefig(plots_dir+'/fkine_errors_x%s.png'%(_suffix), dpi=1200)
    plt.close()
    
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
    plt.savefig(plots_dir+'/fkine_errors_x_dot%s.png'%(_suffix), dpi=1200)
    plt.close()

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
    plt.close()
    return

if __name__ == '__main__':
    model_kwargs_link = dict()
    model_kwargs_link['lr'] = 5e-4#[1e-5, 1e-6, 1e-7]
    model_kwargs_mono = model_kwargs_link.copy()

    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 100
    learn_kwargs['learn_steps'] = 1000 
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['batch_size'] = 10 
    learn_kwargs['n_iter'] = 10
    learn_kwargs['append'] = False 
    learn_kwargs['refine'] = True 
    
    mono_n_hidden = [4, 5, 5, 6, 6, 6]
    mono_s_hidden = [32, 32, 64, 64, 64, 64]

    for n_dims in [3]: #2, 3]:
        for _nj, n_joints in enumerate([7]):#:2, 3, 4, 5, 6, 7]):
            model_kwargs_link['n_dims'] = n_dims 
            model_kwargs_link['n_joints'] = n_joints
            # these are constant for fkine linked
            model_kwargs_link['n_hidden'] = 3
            model_kwargs_link['size_hidden'] = 32
            model_kwargs_link['model'] = 'FKineLinked'
            learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, learn_kwargs, device=device)
    
            model_kwargs_mono['n_dims'] = n_dims 
            model_kwargs_mono['n_joints'] = n_joints
            model_kwargs_mono['n_hidden'] = mono_n_hidden[_nj]
            model_kwargs_mono['size_hidden'] = mono_s_hidden[_nj]
            model_kwargs_mono['model'] = 'FKineMono'
            learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_mono, learn_kwargs, device=device)

            test('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, model_kwargs_mono, device=device)



