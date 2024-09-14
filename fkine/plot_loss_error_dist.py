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
    
    link_model_files_ee = sorted(list(glob(models_dir+".ee"+fkine_link_file+"*.pt")))
    link_model_files_nj = sorted(list(glob(models_dir+".nj"+fkine_link_file+"*.pt")))
    mono_model_files_ee = sorted(list(glob(models_dir+".ee"+fkine_mono_file+"*.pt")))
    mono_model_files_nj = sorted(list(glob(models_dir+".nj"+fkine_mono_file+"*.pt")))

    labels = [r'$\Phi^l$ ee', r'$\Phi^l$ fj', r'$\Phi^m$ ee', r'$\Phi^l$ fj']
    
    for link_model_file_ee, link_model_file_nj, mono_model_file_ee, mono_model_file_nj in zip(link_model_files_ee, link_model_files_nj, mono_model_files_ee, mono_model_files_nj):
        y, y_dot, q, q_dot = [], [], [], []
        _run_link_ee = str.replace(link_model_file_ee, models_dir+".ee"+fkine_link_file, '').replace('.pt', '')
        _run_link_nj = str.replace(link_model_file_nj, models_dir+".nj"+fkine_link_file, '').replace('.pt', '')
        _run_mono_ee = str.replace(mono_model_file_ee, models_dir+".ee"+fkine_mono_file, '').replace('.pt', '')
        _run_mono_nj = str.replace(mono_model_file_nj, models_dir+".nj"+fkine_mono_file, '').replace('.pt', '')
        assert(_run_link_ee == _run_mono_ee)
        assert(_run_link_nj == _run_mono_nj)
        assert(_run_link_nj == _run_mono_ee)
        _run = _run_link_ee

        link_net_ee = FKineLinked(**model_kwargs_link, device=device)
        link_net_ee.load_state_dict(torch.load(link_model_file_ee, map_location=torch.device(device)))
        link_net_ee = link_net_ee.to(device)
        
        link_net_nj = FKineLinked(**model_kwargs_link, device=device)
        link_net_nj.load_state_dict(torch.load(link_model_file_nj, map_location=torch.device(device)))
        link_net_nj = link_net_nj.to(device)

        mono_net_ee = FKineMono(**model_kwargs_mono, device=device)
        mono_net_ee.load_state_dict(torch.load(mono_model_file_ee, map_location=torch.device(device)))
        mono_net_ee = mono_net_ee.to(device)
                                                                                                 
        mono_net_nj = FKineMono(**model_kwargs_mono, device=device)
        mono_net_nj.load_state_dict(torch.load(mono_model_file_nj, map_location=torch.device(device)))
        mono_net_nj = mono_net_nj.to(device)

        env.reset() 
        q, q_dot, y, y_dot = env.get_wrapper_attr('sample_states')(n_samples=n_samples, strategy=sampling_strat)
        
        n = len(y)
        y = np.array(y)
        q = torch.Tensor(q).to(device).requires_grad_(True)
   
        # check the model.eval() functon
        with torch.no_grad():
            y_pred_link_ee, _ = link_net_ee(q)
            y_pred_link_nj, _ = link_net_nj(q)
            y_pred_mono_ee = mono_net_ee(q)
            y_pred_mono_nj = mono_net_nj(q)
        
        palette = itertools.cycle(seaborn.color_palette())
        colors = [next(palette) for i in range(4)]
        
        # Errors
        error_link_ee = ((y_pred_link_ee.cpu() - y).norm(dim=1)/torch.tensor(y).norm()).transpose(0,1)
        error_link_nj = ((y_pred_link_nj.cpu() - y).norm(dim=1)/torch.tensor(y).norm()).transpose(0,1)
        error_mono_ee = ((y_pred_mono_ee.cpu() - y).norm(dim=1)/torch.tensor(y).norm()).transpose(0,1)
        error_mono_nj = ((y_pred_mono_nj.cpu() - y).norm(dim=1)/torch.tensor(y).norm()).transpose(0,1)

        _min_error = min(
                error_link_ee.min(),
                error_link_nj.min(),
                error_mono_ee.min(),
                error_mono_nj.min(),
                )
        _max_error = max(
                error_link_ee.max(),
                error_link_nj.max(),
                error_mono_ee.max(),
                error_mono_nj.max(),
                )

        # Plot errors hists 
        bins = np.linspace(_min_error*0.9, _max_error*1.1, 50)
        fig, axs = plt.subplots(1, n_joints, sharex='all', sharey='all', figsize=(4*n_joints, 4*1))
        for j in range(n_joints):
            ax = axs[j]
            if j == n_joints-1:
                df = pandas.DataFrame({
                    #r'$\Phi^l$ fj': error_link_nj[j],
                    r'$\Phi^l$': error_link_ee[j],
                    r'$\Phi^m$': error_mono_nj[j],
                    #r'$\Phi^m ee$': error_mono_ee[j],
                    })
            else:
                df = pandas.DataFrame({
                    #r'$\Phi^l$ fj': error_link_nj[j],
                    r'$\Phi^l$': error_link_ee[j],
                    r'$\Phi^m$': error_mono_nj[j],
                    })

            g = seaborn.histplot(
                    data=df,
                    bins=bins,
                    palette=colors[0:4],
                    common_norm=False,
                    #multiple='dodge',
                    alpha=0.8,
                    stat='percent',
                    shrink=0.8,
                    ax=ax,
                    legend=(j==n_joints-1),
                    common_bins=False
                    )
            g.set(xlim=(0,_max_error*0.5))
            #seaborn.kdeplot(data=df, palette=colors[0:4], ax=ax, common_norm=True, common_grid=True, legend=(j==n_joints-1), clip=[_min_error*0.9, _max_error*1.1])
            ax.title.set_text('Joint %d'%(j+1))
            ax.set_xlabel(r'Error (\%)')
            if j == 0:
                ax.set_ylabel(r'\# Samples (\%)')
        plt.savefig(plots_dir+'/fkine_errors_hist_x_%s.pdf'%(_run), dpi=1200, bbox_inches='tight')
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
        for j, n_joints in enumerate([7, 6, 5, 4, 3, 2]):
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

            g = seaborn.lineplot(df, ax=ax, alpha=0.8, palette=colors, legend=(j==0 and j==0))
            g.set(yscale='log')
            ax.set_xlabel('Epochs')
            ax.set_ylim([0., 0.7])
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
    learn_kwargs_link['learn_steps'] = 5000 
    learn_kwargs_link['n_envs'] = 32 
    learn_kwargs_link['n_iter'] = 25 
    learn_kwargs_link['append'] = True 
    learn_kwargs_link['refine'] = False 
    learn_kwargs_mono = learn_kwargs_link.copy()
    
    n_runs = 10
    n_samples = 10000

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

            compare('results/fkine_models', 'compare/results', 'compare/plots', model_kwargs_link, model_kwargs_mono, n_samples=n_samples, sampling_strat='random', device=device)
            #break

            print('\n\n\n\n')
        #break
    #plot_loss_comparison('compare/results', 'compare/plots')
    #quit()



