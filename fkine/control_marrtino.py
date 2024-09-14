import time, math, sys
import numpy as np
import torch

# Import our model
from fkine.fkine_linked import FKineLinked
from fkine.fkine_monolithic import FKineMono
from fkine.learn import learn
from fkine.fkine_common import get_hyper_params

# Utility stuff
from utils import * 
from glob import glob

# save video
from gymnasium.wrappers import RecordVideo

# check device
if torch.cuda.is_available():
    device = torch.device(torch.cuda.current_device()) 
    device_name = torch.cuda.get_device_name()
    print("Using GPU %s" %device_name)
else:
    device = 'cpu'
    print("Using CPU")

if __name__ == '__main__':
    seed = 22
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = 'Mono'
    video = True 
    hyperparams_dir = path.cwd()/'results/tunning_fkine'
    models_dir = 'results/marrtino_fkine_models'
    max_steps = 200
    
    n_joints = 5
    n_dims = 3
    
    model_kwargs = dict()
    model_kwargs['model'] = 'FKine'+model
    model_kwargs['n_dims'] = n_dims 
    model_kwargs['n_joints'] = n_joints

    hp_file = sorted(list(hyperparams_dir.glob(f'reacher%dd%dj_FKine{model}_hyperparams.pickle'%(n_dims, n_joints))))
    _, model_params, n_params = get_hyper_params(hp_file[0])
    print('link: ', model_params, n_params)
    model_kwargs.update(model_params)

    env = make_env(
            render_mode="rgb_array" if video else "human",
            model_file=path.cwd()/'rgym/envs/assets/marrtino.xml',
            width=1440,
            height=720
            ) 
    
    if video:
        env = RecordVideo(env, video_folder='./videos', name_prefix=model, episode_trigger=lambda x: True)

    obs = env.reset()

    # Load Fkine model
    model_kwargs['model'] = None
    _suffix = model_kwargs_2_str(**model_kwargs)
    fkine_file = f"/fkine_FKine{model}"+_suffix
    fkine_model_file = sorted(list(glob(models_dir+fkine_file+"*.pt")))[0]
    print('model file: ', fkine_model_file)

    fkine_net = eval(f'FKine{model}')(**model_kwargs, device=device)
    fkine_net.load_state_dict(torch.load(fkine_model_file, map_location=torch.device(device)))
    fkine_net = fkine_net.to(device)

    stiffness = torch.tensor([5, 5, 5], dtype=torch.float)
    dumping = torch.tensor([1, 1, 1], dtype=torch.float)
    K = torch.diag(stiffness).to(device)
    D = torch.diag(dumping).to(device) 
    target = torch.tensor(env.get_wrapper_attr('get_body_com')("target")[:n_dims], dtype=torch.float).reshape(-1,1).to(device) 
    #print('target: ', target)
    target_vel = torch.zeros(size=(n_dims, 1), dtype=torch.float, device=device)

    for step in range(max_steps):
        #print('step: ', step)
        obs = env.get_wrapper_attr('get_observation')()
        #print('obs: ', obs)
        # we dont need x and xdot since we estimate it from fkine_net
        #x = torch.tensor(obs['x'][:,-1:n_joints], dtype=torch.float, device=device)
        #xdot = torch.tensor(obs['xdot'][:, -1:n_joints], dtype=torch.float, device=device)
        q = torch.tensor(obs['q'], dtype=torch.float, device=device)
        qdot = torch.tensor(obs['qdot'], dtype=torch.float, device=device).reshape(-1,1)
        #print('q: ', q)
        #print('qdot: ', qdot)
        
        with torch.no_grad():
            if model == 'Linked':
                _pred, _ = fkine_net(q)
                try:
                    _J, _ = torch.autograd.functional.jacobian(fkine_net, q)
                except:
                    print('Autograd FAILED to compute derivative')
                    _J = torch.zeros(1, n_dims, n_joints, n_joints, dtype=torch.float, device=device)
            else:
                _pred = fkine_net(q)
                try:
                    _J = torch.autograd.functional.jacobian(fkine_net, q)
                except:
                    print('Autograd FAILED to compute derivative')
                    _J = torch.zeros(1, n_dims, n_joints, n_joints, dtype=torch.float, device=device)
        # first 0 is because we only have one sample
        # second dim is n_dims (x, y, z)
        # third dim is the joint in cartesia (x0, ..., x4)
        # last dim is the joint angle (theta0, ..., theta4)
        # we want the derivative of the end effector (last third dim) w.r.t. all angles 
        jacob = _J[0,:,-1,:]
        #print('jacob: ', jacob)

        # predict current x and xdot using fkine_net
        x_pred = _pred[0,:,-1:n_joints] 
        #print('xpred: ', x_pred)
        xdot_pred = jacob@qdot 
        #print('xdot pred: ', xdot_pred)
        #print('xdot: ', xdot)

        # compute external force
        fext = K@(target-x_pred) + D@(target_vel-xdot_pred)
        #print('fext: ', fext)

        #print('J: ', _J)
        torque = jacob.T@fext
        #print('torque: ', torque)

        # send it
        env.step(torque.flatten().detach().cpu().numpy())
        if not video:
            env.render()
            if model == 'Mono':
                time.sleep(0.1)
    env.close()
