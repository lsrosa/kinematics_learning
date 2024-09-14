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
            print('sample: ', i)
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
        
        fig, axs = plt.subplots(n_dims, n_joints, sharex='all', figsize=(4*n_joints+0.5, 3*n_dims))
        for d in range(n_dims):
            for j in range(n_joints):
                idx = j+d*n_joints
                ax = axs[d, j]
                df = pandas.DataFrame({r'$\Phi^l$': y_pred_plot_link[:,idx], r'$\Phi^m$': y_pred_plot_mono[:,idx], r'G.T.': y_plot[:,idx]})
                seaborn.lineplot(df, ax=ax, alpha=0.8, palette=colors, legend=(d==0 and j==0))
                if d==0: ax.title.set_text('Joint %d'%(j+1))
                ax.set_xlabel('Step')
                ax.set_ylabel(_y_labels[idx])
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
            ax.set_ylabel(_y_labels[d])
        plt.savefig(plots_dir+'/fkine_predictions_xdot_%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close() 
        
        # Plot errors histogram
        error_joints_link = (y_pred_link.cpu() - y).norm(dim=1).transpose(0,1)
        error_joints_mono = (y_pred_mono.cpu() - y).norm(dim=1).transpose(0,1)
        _min_error = min(error_joints_link.min(), error_joints_mono.min())
        _max_error = max(error_joints_link.max(), error_joints_mono.max())
        bins = np.linspace(_min_error*0.9, _max_error*1.1, 50)
        fig, axs = plt.subplots(1, n_joints, sharex='all', sharey='all', figsize=(4*n_joints, 4*1))
        for j in range(n_joints):
            ax = axs[j]
            df = pandas.DataFrame({r'$\Phi^l$': error_joints_link[j], r'$\Phi^m$': error_joints_mono[j]})
            seaborn.histplot(data=df, bins=bins, palette=colors[0:2], common_norm=False, stat='percent', shrink=0.8, ax=ax, legend=(j==0), common_bins=False, kde=False, kde_kws={'clip':[_min_error*0.9, _max_error*1.1], 'cut':10})
            ax.title.set_text('Joint %d'%(j+1))
            ax.set_xlabel('Error')
            if j == 0:
                ax.set_ylabel(r'\# Samples (\%)')
        plt.savefig(plots_dir+'/fkine_errors_hist_x_%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close() 
        
        # plot errors correlations
        fig, axs = plt.subplots(1, 1, sharex='all', sharey='all', figsize=(4, 4))
        _x_corr_label = r'$<||x_{%d} - \tilde{x}_{%d}||>$'%(0, 0)
        _phi_link_label = [r'$\Phi^l$' for i in range(n_samples)]
        _phi_mono_label = [r'$\Phi^m$' for i in range(n_samples)]
        
        #normalize
        _error_joints_link_norm = error_joints_link[0] - error_joints_link[0].min()
        _error_joints_link_norm /= _error_joints_link_norm.max()
        _error_joints_mono_norm = error_joints_mono[0] - error_joints_mono[0].min()
        _error_joints_mono_norm /= _error_joints_mono_norm.max()
        _x_corr = np.hstack((_error_joints_link_norm, _error_joints_mono_norm))
        df = pandas.DataFrame({
            'Arch': _phi_link_label+_phi_mono_label,
            _x_corr_label : _x_corr,
            })
        ax = axs
        _y_corr_label = r'$<||x_{%d} - \tilde{x}_{%d}||>$'%(j+1, j+1)
        _error_joints_link_norm = error_joints_link[n_joints-1] - error_joints_link[n_joints-1].min()
        _error_joints_link_norm /= _error_joints_link_norm.max()
        _error_joints_mono_norm = error_joints_mono[n_joints-1] - error_joints_mono[n_joints-1].min()
        _error_joints_mono_norm /= _error_joints_mono_norm.max()
        _y_corr = np.hstack((_error_joints_link_norm, _error_joints_mono_norm))
        df[_y_corr_label] = _y_corr 
        seaborn.scatterplot(data=df, ax=ax, x=_x_corr_label, y=_y_corr_label, hue='Arch', palette=colors[0:2], alpha=0.5)
        plt.savefig(plots_dir+'/fkine_errors_correlations_%s%s.pdf'%(_suffix, _run), dpi=1200, bbox_inches='tight')
        plt.close() 

    return

def plot_loss_comparison(results_dir, plots_dir):
    results_dir = path(results_dir)
    plots_dir = path(plots_dir)
    plt.rcParams['text.usetex'] = True
    
    fig, axs = plt.subplots(1, 6, sharex='all', sharey='all', figsize=(4*6, 3*1))
    palette = itertools.cycle(seaborn.color_palette())
    colors = [next(palette) for i in range(2)]
    
    for d, n_dims in enumerate([3]):#, 2]):
        for j, n_joints in enumerate([2, 3, 4, 5, 6, 7]):
            loss_link_file = sorted(list(results_dir.glob("fkine_FKineLinked_%dd%dj_*.pickle"%(n_dims, n_joints))))
            loss_mono_file = sorted(list(results_dir.glob("fkine_FKineMono_%dd%dj_*.pickle"%(n_dims, n_joints))))
            if (not loss_link_file) or (not loss_mono_file):
                print('cannot find losses file for', loss_link_file, loss_mono_file, '. Skipping.')
                continue

            with open(loss_link_file[0].as_posix(), 'rb') as h:
                losses_link, duration = pickle.load(h)
            with open(loss_mono_file[0].as_posix(), 'rb') as h:
                losses_mono, duration = pickle.load(h)
            
            n_runs = losses_link.shape[0]
            n_epochs = losses_link.shape[1]
            epochs = np.linspace(1, n_epochs, n_epochs)
            ax = axs[j]

            mean_link = np.mean(losses_link, axis=0)
            mean_mono = np.mean(losses_mono, axis=0)
            df = pandas.DataFrame({r'$\Phi^l$': mean_link, r'$\Phi^m$': mean_mono})

            seaborn.lineplot(df, ax=ax, alpha=0.8, palette=colors, legend=(j==0 and j==0))
            ax.set_xlabel('Epochs')
            if n_joints==2: 
                ax.set_ylabel(r'$\mathcal{L}$')
            
            std_link = 2*np.std(losses_link, axis=0)
            std_mono = 2*np.std(losses_mono, axis=0)
            ax.fill_between(epochs, mean_link-std_link, mean_link+std_link, alpha=0.3)
            ax.fill_between(epochs, mean_mono-std_mono, mean_mono+std_mono, alpha=0.3)
            ax.title.set_text('%d-Joints'%(j+2))
    plt.savefig(plots_dir/'fkine_link_x_mono.pdf', dpi=1200, bbox_inches='tight')
    plt.close()
    return

if __name__ == '__main__':
    hyperparams_dir = path.cwd()/'results/tunning_fkine'
    model_kwargs_link = dict()
    model_kwargs_mono = model_kwargs_link.copy()

    learn_kwargs_link = dict()
    learn_kwargs_link['seed'] = 1
    learn_kwargs_link['n_rollouts'] = 100
    learn_kwargs_link['learn_steps'] = 3000 
    learn_kwargs_link['n_envs'] = 32 
    learn_kwargs_link['n_iter'] = 25 
    learn_kwargs_link['append'] = False 
    learn_kwargs_link['refine'] = True 
    learn_kwargs_mono = learn_kwargs_link.copy()
    
    n_runs = 5

    for n_dims in [3]:#, 2]:
        for _nj, n_joints in enumerate([7]):#, 6, 5, 4, 3, 2]):
            print('dims: ', n_dims, '   joints: ', n_joints)
            hp_file_link = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineLinked_hyperparams.pickle'%(n_dims, n_joints))))
            hp_file_mono = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineMono_hyperparams.pickle'%(n_dims, n_joints))))
            if (not hp_file_link):
                if not hp_file_link: print('could not find fkineLinked hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
                #continue
                
            if (not hp_file_mono):
                if not hp_file_mono: print('could not find fkineMono hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
                continue

            model_kwargs_link['model'] = 'FKineLinked'
            model_kwargs_link['n_dims'] = n_dims 
            model_kwargs_link['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_link[0])
            print('link: ', learn_params, model_params, n_params)
            model_kwargs_link.update(model_params)
            learn_kwargs_link.update(learn_params)
            for run in range(n_runs):
                learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, learn_kwargs_link, device=device)
    
            model_kwargs_mono['model'] = 'FKineMono'
            model_kwargs_mono['n_dims'] = n_dims 
            model_kwargs_mono['n_joints'] = n_joints
            learn_params, model_params, n_params = get_hyper_params(hp_file_mono[0])
            print('mono: ', learn_params, model_params, n_params)
            model_kwargs_mono.update(model_params)
            learn_kwargs_mono.update(learn_params)
            for run in range(n_runs):
                learn('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_mono, learn_kwargs_mono, device=device)

            compare('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, model_kwargs_mono, n_samples=100, sampling_strat='walk', device=device)
            #break

            print('\n\n\n\n')
        #break
    plot_loss_comparison('compare/results', 'compare/plots')
    #quit()



