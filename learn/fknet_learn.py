import math, time, argparse
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import gymnasium as gym
import rgym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from networks import *


def get_info(env_name):
    fknet_file = None
    n_joints = 0
    n_out = 0
    if env_name[0:10] == 'AbsReacher':
        fknet_file = "fknet_AbsReacher.pth"
        n_joints = 2
        n_out = 2 # output dimension
    if env_name[0:6] == 'Pusher':
        fknet_file = "fknet_Pushher.pth"
        n_joints = 7
        n_out = 3
    else:
        assert False, f"Env {env_name} not supported."
    return  fknet_file, n_joints, n_out


def learn(env_name, seed=2000, learn_steps=100000, random_steps=True):

    fknet_file, n_joints, n_out = get_info(env_name)

    num_envs = 100
    env = make_vec_env(env_name, n_envs=num_envs)

    fknet_file = "fknet_Pusher.pth"

    fknet = FKNet(n_in=n_joints, n_out=n_out)
    try:
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)

    model_cb = ModelLearnCB()
    model_cb.env = env
    model_cb.fknet = fknet

    current_seed = seed

    try:
        if random_steps:
            a = np.zeros((num_envs,env.action_space.shape[0]))
            istep = 0
            run = True
            while istep<learn_steps and run:
                print(f"{istep:6d} | Training fknet on {env_name} with seed {current_seed} ...")
                env.seed(seed=current_seed)
                current_seed += num_envs
                obs = env.reset()
                model_cb._on_rollout_start()
                for _ in range(100):
                    for ienv in range(env.num_envs):
                        a[ienv] = env.envs[ienv].action_space.sample()
                    obs,_,_,_ = env.step(a)
                    istep += num_envs
                    run = model_cb._on_step()
                model_cb._on_rollout_end()
                # print(" -- rollout end --")

        else:
            model = SAC("MlpPolicy", env, gradient_steps=2,
                seed=seed, verbose=0)

            print(f"Training model with seed {seed} for {learn_steps} steps ...")

            model.learn(total_timesteps=learn_steps,
                callback = model_cb,
                )
            del model

    except KeyboardInterrupt:
        print("User quit!")




    fknet.save(fknet_file)
    print(f"Saved FKnet {fknet_file}")

    env.close()


def one_run(env, fknet, verbose=0):

    tol = 0.075

    obs, _ = env.reset()

    x0 = obs[0:fknet.n_in]
    x = x0
    # Reacher
    #t = obs[2:4]
    # Pusher
    t = obs[2*fknet.n_in+fknet.n_out:2*fknet.n_in+2*fknet.n_out]


    if verbose>0:
        print(f"x0 {vstr(x)} tt {vstr(t)}")

    xt = fknet.inverse(x,t,tol=tol*0.9,iters=30000)
    tt = fknet.forward(xt).detach().numpy()
    xt = xt.detach().numpy()
    d = np.linalg.norm(t-tt)
    if verbose>0:
        print(f"x0 {vstr(x0)} -> xtarget {vstr(xt)} -> ttarget {vstr(tt)} | {d} ")


    for i in range(100):

        x = obs[0:fknet.n_in]

        # Reacher
        #v = obs[4:6]
        #y = env.unwrapped.get_body_com("fingertip")[0:2]

        # Pusher
        v = obs[fknet.n_in:2*fknet.n_in]
        y = env.unwrapped.get_body_com("tips_arm")

        y_pred = fknet.predict(x).detach().numpy()

        err = y_pred - y

        dy = y_pred - t
        dx = x - xt

        g = fknet.derivative(x,dy)
        #xt = fknet.inverse(x,t)

        k1 = 5
        vdes = - k1 * dx

        dt = 0.2
        a = (vdes - v) * dt

        a = np.clip(a,-1,1)

        reached = np.linalg.norm(dy)<tol
        reached_str = "******" if reached else ""
        if verbose>0:
            print(f"x {vstr(x)} | v {vstr(v)} | ee {vstr(y)} | pred {float(y_pred[0]):6.3f} {float(y_pred[1]):6.3f} | err {float(err[0]):6.3f} {float(err[1]):6.3f} | dx {float(dx[0]):6.3f} {float(dx[1]):6.3f} | dy {float(dy[0]):6.3f} {float(dy[1]):6.3f} | grad {float(g[0]):6.3f} {float(g[1]):6.3f} | vdes {float(vdes[0]):6.3f} {float(vdes[1]):6.3f} | a {float(a[0]):6.3f} {float(a[1]):6.3f} {reached_str}")

        obs,_,_,_,_ = env.step(a)

    if not reached:
        print(f"Failed | x0 {vstr(x0)} | t {vstr(t)} | ee {vstr(y)} | x {vstr(x)} | d {vstr(dy)} -> {vstr(np.linalg.norm(dy))} ")

    return reached


def test(env_name, fknet_file, render=False):

    _, n_joints, n_out = get_info(env_name)

    fknet = FKNet(n_in=n_joints, n_out=n_out)
    try:
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)
        return

    env = gym.make(env_name, render_mode="human" if render else None)

    one_run(env,fknet,verbose=1)

    '''
    if render:
        for i in range(30):
            env.render()
            time.sleep(0.1)
    '''


def test2(atarget=None, ttarget=None, astart=None, render=False, verbose=0):

    fknet = FKNet(n_in=2, n_out=2)
    try:
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)
        return

    env = gym.make("AbsReacher6v", render_mode="human" if render else None)

    if atarget is not None:
        xt = atarget
        tt,tt_l1 = env.unwrapped.fk(xt)
        ttp = fknet.predict(xt)
    elif ttarget is not None:
        xt = [-999,-999]
        tt = ttarget
        tt_l1 = None
        ttp = tt

    if astart is not None:
        x0 = astart
    else:
        x0 = [ 0.0, 0.0 ]
    t0,t0_l1 = env.unwrapped.fk(x0)
    t0p = fknet.predict(x0)
    dt = np.array(tt)-np.array(t0)
    g = fknet.derivative(x0,dt)
    print(f" x0 {vstr(x0)} | t0' {vstr(t0p)} | t0 {vstr(t0)}")
    print(f" xt {vstr(xt)} | tt' {vstr(ttp)} | tt {vstr(tt)}")
    print(f" x0 {vstr(x0)} | xt {vstr(xt)} | t0p {vstr(t0p)} | tt {vstr(tt)} | dt {vstr(dt)} | g {vstr(g)}")

    ixt = fknet.inverse(x0,tt,iters=1000,verbose=1)
    itt,itt_l1 = env.unwrapped.fk(ixt)

    print(f" xt {vstr(xt)} | ixt {vstr(ixt)} ")

    if render:
        fig, ax = plt.subplots()
        ax.set_xlim(-0.25, 0.25)
        ax.set_ylim(-0.25, 0.25)
        ax.plot(t0[0], t0[1], 'o', color='tab:brown')
        ax.plot(tt[0], tt[1], 'o', color='tab:red')
        ax.plot(itt[0], itt[1], 'o', color='tab:green')
        ax.plot(0,0, '+', color=(0,0,0))
        #ax.plot([0,t0_l1[0]], [0, t0_l1[1]], color=(0,0,0))
        #ax.plot([t0_l1[0],t0[0]], [t0_l1[1], t0[1]], color=(0,0,0))
        if tt_l1 != None:
            ax.plot([0,tt_l1[0]], [0, tt_l1[1]], color=(0,0,0))
            ax.plot([tt_l1[0],tt[0]], [tt_l1[1], tt[1]], color=(0,0,0))
        ax.plot([0,itt_l1[0]], [0, itt_l1[1]], color=(0,0,0))
        ax.plot([itt_l1[0],itt[0]], [itt_l1[1], itt[1]], color=(0,0,0))

        plt.show()

    env.close()

    env = gym.make("ReacherMod6vSRFT", render_mode="human" if render else None)

    env.set_fixed_target(tt)

    one_run(env,fknet,verbose=0)

    env.close()



def eval_fknet(env_name, fknet_file, n=100):

    _, n_joints, n_out = get_info(env_name)

    env = gym.make(env_name)

    # env = gym.make("ReacherMod6vSRFT")
    # env.set_fixed_target([0.0303, -0.1172])

    fknet = FKNet(n_in=n_joints, n_out=n_out)
    try:
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)
        return

    print(f"Evaluating FKNet {fknet_file} for {n} episodes")
    r = 0
    try:
        for i in tqdm(range(n)):
            r += one_run(env,fknet,verbose=0)
        i += 1
    except:
        pass
    print(f"Success rate = {r}/{i} = {r/i*100:.2f} %")



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-env', type=str, default = 'AbsReacher6v',
        help='Env name')
    parser.add_argument("-fknet", type=str, default=None,
        help="FKNet model (default: None)")
    parser.add_argument('--test', default = False, action ='store_true',
        help='Test target')
    parser.add_argument('-astart', type=float, nargs=2, default = None,
        help='Test start angle')
    parser.add_argument('-atarget', type=float, nargs=2, default = None,
        help='Test target angle')
    parser.add_argument('-ttarget', type=float, nargs=2, default = None,
        help='Test target position')
    parser.add_argument('-eval', type=int, default = None,
        help='Eval iterations')
    parser.add_argument('--render', default = False, action ='store_true',
        help='Render')

    args = parser.parse_args()

    if args.test:
        test(args.env,args.fknet,args.render)
    elif args.atarget is not None or args.ttarget is not None:
        test2(atarget=args.atarget, ttarget=args.ttarget, astart=args.astart, render=args.render)
    elif args.eval is not None:
        eval_fknet(args.env,args.fknet,args.eval)
    else:
        learn(args.env,seed=2000, learn_steps=100000)

