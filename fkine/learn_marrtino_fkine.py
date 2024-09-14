from PIL import Image
import time, math, sys
import numpy as np
import torch

# Import ouur model
from fkine.fkine_linked import FKineLinked
from fkine.learn import learn
from fkine.fkine_common import get_hyper_params

# Utility stuff
from utils import * 
from glob import glob

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

def get_cart_poses(models_dir, plots_dir, model_kwargs, device='cpu'):
    qs = np.array([
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0., np.pi/2, np.pi/2, -np.pi/2, 0.],
        [np.pi/4, np.pi/2, np.pi/2, -np.pi/2, np.pi/4]
        ])
    
    n_joints = 5
    n_dims = 3
    env = make_env(
            render_mode="rgb_array",
            model_file=path.cwd()/'rgym/envs/assets/marrtino.xml',
            width=2880,
            height=1440
            ) 
    obs = env.reset()
    # Load Fkine model
    model_kwargs['model'] = None
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = "/fkine_FKineLinked"+_suffix
    fkine_model_file = sorted(list(glob(models_dir+fkine_file+"*.pt")))[0]
    print('model file: ', fkine_model_file)

    fkine_net = FKineLinked(**model_kwargs, device=device)
    fkine_net.load_state_dict(torch.load(fkine_model_file, map_location=torch.device(device)))
    fkine_net = fkine_net.to(device)

    # Define a set of poses
    qdot = np.zeros(n_joints+3,)
    arrs = []
    for q in qs:
        _q = torch.Tensor(q).to(device)
        with torch.no_grad():
            _y_pred, _ = fkine_net(_q)
        
        print('\n\n--------\n')
        print('q: ', q)
        print('x pred (cm): ', 100*_y_pred[0,:,-1].cpu().detach().numpy())
        
        # save images of those poses
        env.get_wrapper_attr('set_state')(np.hstack([q,[0., 0., 0.]]), qdot)
        arrs.append(env.render())
    
    ssdir = path.cwd()/plots_dir
    ssdir.mkdir(parents=True, exist_ok=True)
    
    for i, arr in enumerate(arrs):
        im = Image.fromarray(arr)
        im.save(ssdir.as_posix()+'/marrtino%d.jpeg'%i)
    env.close()


if __name__ == '__main__':
    hyperparams_dir = path.cwd()/'results/tunning_fkine'

    learn_kwargs = dict()
    learn_kwargs['seed'] = 1
    learn_kwargs['n_rollouts'] = 100
    learn_kwargs['learn_steps'] = 3000 
    learn_kwargs['n_envs'] = 32 
    learn_kwargs['n_iter'] = 25 
    learn_kwargs['append'] = False 
    learn_kwargs['refine'] = True 
    
    n_runs = 5
    n_dims = 3
    n_joints = 5
    
    print('dims: ', n_dims, '   joints: ', n_joints)
    hp_file = sorted(list(hyperparams_dir.glob('reacher%dd%dj_FKineLinked_hyperparams.pickle'%(n_dims, n_joints))))

    model_kwargs = dict()
    model_kwargs['model'] = 'FKineLinked'
    model_kwargs['n_dims'] = n_dims 
    model_kwargs['n_joints'] = n_joints
    
    learn_params, model_params, n_params = get_hyper_params(hp_file[0])
    print('link: ', learn_params, model_params, n_params)
    model_kwargs.update(model_params)
    learn_kwargs.update(learn_params)

    env_kwargs = {'model_file': (path.cwd()/'rgym/envs/assets/marrtino.xml').as_posix()}
    learn('results/marrtino_fkine_models', 'compare/marrtino_results', 'compare/marrtino_plots', model_kwargs, learn_kwargs, env_kwargs, device=device)
    
    get_cart_poses('results/marrtino_fkine_models', 'results/marrtino_plots', model_kwargs, device=device)

