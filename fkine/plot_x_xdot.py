import time, math, sys
import numpy as np
import torch

from stable_baselines3.common.env_util import make_vec_env

# Import ouur model
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from fkine.learn import learn
from fkine.fkine_common import get_hyper_params

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

def compare(models_dir, results_dir, plots_dir, model_kwargs_link, model_kwargs_mono, n_samples=100, sampling_strat='walk', device='cpu'):
    plt.rcParams['text.usetex'] = True

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
    
    fkine_link_model_files = sorted(list(glob(models_dir+fkine_link_file+"*.pt")))
    fkine_mono_model_files = sorted(list(glob(models_dir+fkine_mono_file+"*.pt")))
    
    error_y_link = []
    error_y_mono = []
    error_y_dot_link = []
    error_y_dot_mono = []
    
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
        q, q_dot, y, y_dot = env.get_wrapper_attr('sample_states')(n_samples=n_samples, strategy=sampling_strat)
        
        n = len(y)
        y = np.array(y)
        q_link = torch.Tensor(q).to(device).requires_grad_(True)
        q_mono = torch.Tensor(q).to(device).requires_grad_(True)
        q_dot = torch.Tensor(q_dot).to(device)
   
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
        
        palette = itertools.cycle(seaborn.color_palette())
        colors = [next(palette) for i in range(3)]
        
        _y_labels = []
        for d in range(n_dims):
            for j in range(n_joints):
                _y_labels += [r'$x_{%d,%d}$'%(j,d)]
        _suffix = '%dd%dj'%(n_dims, n_joints) 
        
        fig, axs = plt.subplots(n_dims, n_joints, sharey="all", sharex='all', figsize=(4*n_joints+0.5, 3*n_dims))
        for d in range(n_dims):
            for j in range(n_joints):
                idx = j+d*n_joints
                ax = axs[d, j]
                df = pandas.DataFrame({r'$\Phi^l$': y_pred_plot_link[:,idx], r'$\Phi^m$': y_pred_plot_mono[:,idx], r'G.T.': y_plot[:,idx]})
                seaborn.lineplot(df, ax=ax, alpha=0.8, palette=colors, legend=(d==0 and j==0))
                if d==0: ax.title.set_text('Joint %d'%(j+1))
                ax.set_xlabel('Step')
                ax.set_ylabel(_y_labels[idx]+' (m)')
        plt.savefig(plots_dir+'/fkine_predictions_x_%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close()

        _y_labels = []
        for d in range(n_dims):
            _y_labels += [r'$\dot{x}_{%d,%d}$'%(n_joints, d)]

        fig, axs = plt.subplots(n_dims, 1, sharex='all', figsize=(4*1, 4*1))
        for d in range(n_dims):
            ax = axs[d]
            df = pandas.DataFrame({r'$\Phi^l$': y_dot_pred_plot_link[:,d], r'$\Phi^m$': y_dot_pred_plot_mono[:,d], r'G.T.': y_dot_plot[:,d]})
            seaborn.lineplot(df, ax=ax, alpha=0.8, palette=colors, legend=(d==0))
            if d==0: seaborn.move_legend(ax, "lower center", ncol=3, title=None, bbox_to_anchor=(.5, 1))
            ax.set_xlabel('Step')
            ax.set_ylabel(_y_labels[d]+' (m/s)')
        plt.savefig(plots_dir+'/fkine_predictions_xdot_%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close() 
    return

if __name__ == '__main__':
    hyperparams_dir = path.cwd()/'results/tunning_fkine'
    model_kwargs_link = dict()
    model_kwargs_mono = model_kwargs_link.copy()

    learn_kwargs_link = dict()
    learn_kwargs_link['seed'] = 1
    learn_kwargs_link['n_rollouts'] = 100
    learn_kwargs_link['learn_steps'] = 5000 
    learn_kwargs_link['n_envs'] = 32 
    learn_kwargs_link['n_iter'] = 25 
    learn_kwargs_link['append'] = True 
    learn_kwargs_link['refine'] = False 
    learn_kwargs_mono = learn_kwargs_link.copy()
    
    n_runs = 10
    n_samples = 100

    for n_dims in [3]:#, 2]:
        for _nj, n_joints in enumerate([7]): #[7, 6, 5, 4, 3, 2]):
            print('dims: ', n_dims, '   joints: ', n_joints)
            hp_file_link = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineLinked_hyperparams.pickle'%(n_dims, n_joints))))
            hp_file_mono = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineMono_hyperparams.pickle'%(n_dims, n_joints))))

            model_kwargs_link['model'] = 'FKineLinked'
            model_kwargs_link['n_dims'] = n_dims 
            model_kwargs_link['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_link[0])
            print('link: ', learn_params, model_params, n_params)
            model_kwargs_link.update(model_params)
            learn_kwargs_link.update(learn_params)
    
            model_kwargs_mono['model'] = 'FKineMono'
            model_kwargs_mono['n_dims'] = n_dims 
            model_kwargs_mono['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_mono[0])
            print('mono: ', learn_params, model_params, n_params)
            model_kwargs_mono.update(model_params)
            learn_kwargs_mono.update(learn_params)

            compare('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, model_kwargs_mono, n_samples=n_samples, sampling_strat='walk', device=device)
            #break


