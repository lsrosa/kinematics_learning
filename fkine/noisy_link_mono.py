import time, math, sys
import numpy as np
import torch
import re

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

def compare_noise(models_dir, results_dir, plots_dir, model_kwargs_link, model_kwargs_mono, n_samples=100, sampling_strat='walk', device='cpu'):
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
    
    # plot defs
    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    df = pandas.DataFrame({'sigma':[], 'model': [], 'error': []})
    
    for fkine_link_model_file, fkine_mono_model_file in zip(fkine_link_model_files, fkine_mono_model_files):
        y, y_dot, q, q_dot = [], [], [], []
        _sigma_link = float(re.findall(r'\d+\.\d+', str.replace(fkine_link_model_file, models_dir+fkine_link_file, '').replace('.pt', ''))[0])
        _sigma_mono = float(re.findall(r'\d+\.\d+', str.replace(fkine_mono_model_file, models_dir+fkine_mono_file, '').replace('.pt', ''))[0])
        assert _sigma_link == _sigma_mono
        _sigma = _sigma_link

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
   
        # check the model.eval() functon
        with torch.no_grad():
            y_pred_link, _ = fkine_link_net(q_link)
            y_pred_mono = fkine_mono_net(q_mono)
        
        palette = itertools.cycle(seaborn.color_palette())
        colors = [next(palette) for i in range(3)]
        
        if _sigma == 0.0:
            print('sigma 0')
            avg_error0_link = ((y_pred_link.cpu() - y).norm(dim=1).transpose(0,1))[-1].mean()
            avg_error0_mono = ((y_pred_mono.cpu() - y).norm(dim=1).transpose(0,1))[-1].mean()
        else:
            error_joints_link = (y_pred_link.cpu() - y).norm(dim=1).transpose(0,1)
            error_joints_mono = (y_pred_mono.cpu() - y).norm(dim=1).transpose(0,1)
            
            sigmas = [_sigma for i in range(n)]

            models = [r'$\Phi^l$' for i in range(n)] 
            df = pandas.concat([df, pandas.DataFrame({'sigma': sigmas, 'model': models, 'error' : error_joints_link[-1]/avg_error0_link})])
            
            models = [r'$\Phi^m$' for i in range(n)] 
            df = pandas.concat([df, pandas.DataFrame({'sigma': sigmas, 'model': models, 'error' : error_joints_mono[-1]/avg_error0_mono})])
    
    df.sort_values(['model','sigma'], ascending=[False, True])
    seaborn.lineplot(data=df, x='sigma', y='error', hue='model', err_style='bars', palette=colors[0:2], ax=ax, legend=True, alpha=0.8)
     
    ax.set_xlabel(r'$\sigma$')
    ax.set_ylabel(r'Error')
    seaborn.move_legend(ax, "upper center", ncol=2, title=None)
    #plt.xticks(rotation=70)
    plt.savefig(plots_dir+'/fkine_errors_variance.pdf', dpi=1200, bbox_inches='tight')
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
    
    n_runs = 1
    n_dims = 3
    n_joints = 7


    hp_file_link = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineLinked_hyperparams.pickle'%(n_dims, n_joints))))
    hp_file_mono = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineMono_hyperparams.pickle'%(n_dims, n_joints))))
    if (not hp_file_link):
        if not hp_file_link: print('could not find fkineLinked hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
        #continue
        
    if (not hp_file_mono):
        if not hp_file_mono: print('could not find fkineMono hyper-parameters for %dd%dj, skipping'%(n_dims, n_joints))
        quit() 

    for noise_var in [0.0, 0.0001, 0.00025, 0.00050, 0.00075, 0.001, 0.0025, 0.0050, 0.0075, 0.01]:
        print('\n\nNoise Var: %f\n\n'%noise_var)

        model_kwargs_link['model'] = 'FKineLinked'
        model_kwargs_link['n_dims'] = n_dims 
        model_kwargs_link['n_joints'] = n_joints
        learn_params, model_params, n_params = get_hyper_params(hp_file_link[0])
        #print('link: ', learn_params, model_params, n_params)
        model_kwargs_link.update(model_params)
        learn_kwargs_link.update(learn_params)
        #for run in range(n_runs):
        #    learn('results/noise_fkine_models', 'compare/noise_results', 'compare/noise_plots', model_kwargs_link, learn_kwargs_link, device=device, noisy=True, noise_var=noise_var)
        
        model_kwargs_mono['model'] = 'FKineMono'
        model_kwargs_mono['n_dims'] = n_dims 
        model_kwargs_mono['n_joints'] = n_joints
        learn_params, model_params, n_params = get_hyper_params(hp_file_mono[0])
        #print('mono: ', learn_params, model_params, n_params)
        model_kwargs_mono.update(model_params)
        learn_kwargs_mono.update(learn_params)
        #for run in range(n_runs):
        #    learn('results/noise_fkine_models', 'compare/noise_results', 'compare/noise_plots', model_kwargs_mono, learn_kwargs_mono, device=device, noisy=True, noise_var=noise_var)

    compare_noise('results/noise_fkine_models', 'compare/noise_results', 'compare/noise_plots', model_kwargs_link, model_kwargs_mono, n_samples=1000, sampling_strat='random', device=device)
    #break




