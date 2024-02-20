import math, time, argparse
import numpy as np

import gymnasium as gym
import rgym

from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env

from networks import *

env_name = 'AbsReacher6v'

fknet_file = "fknet_AbsReacher.pth"
num_envs = 100

def learn(seeds=[1999], learn_steps=100000):

    env = make_vec_env(env_name, n_envs=num_envs) 

    fknet = FKNet(n_in=2, n_out=2)
    try:            
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)

    model_cb = ModelLearnCB()
    model_cb.env = env
    model_cb.fknet = fknet

    for seed in seeds:
        model = SAC("MlpPolicy", env, gradient_steps=2, 
                seed=seed, verbose=0)

        try:
            print(f"Training model with seed {seed} for {learn_steps} steps ...")

            model.learn(total_timesteps=learn_steps,
                callback = model_cb,
                )
        except KeyboardInterrupt:
            print("User quit!")
            break

        del model
        
        
    fknet.save(fknet_file)
    print(f"Saved FKnet {fknet_file}")

    env.close()


def test(render=False):

    fknet = FKNet(n_in=2, n_out=2)
    try:            
        fknet.load(fknet_file)
        print(f"Loaded FKnet {fknet_file}")
    except Exception as e:
        print(e)
        return

    env = gym.make("ReacherMod6v", render_mode="human" if render else None)

    obs, _ = env.reset()

    t = obs[2:4]  

    print(f"t {float(t[0]):6.3f} {float(t[1]):6.3f}")
    
    for i in range(100):

        x = obs[0:2]
        v = obs[4:6]
        y = env.unwrapped.get_body_com("fingertip")[0:2]

        y_pred = fknet.predict(x).detach().numpy()

        err = y_pred - y

        d = y_pred - t

        grad = np.zeros((2,))

        k1 = np.array([1000.0,1000.0])
        k2 = np.array([0.2,1.0])

        if np.linalg.norm(d)<0.02:
            print("************")
            vdes = np.zeros((2,))
        else:
            grad = fknet.derivative(x,t).detach().numpy()
            vdes = k1 * grad * d

        a = k2 * (vdes - v)

        a = np.clip(a,-1,1)

        print(f"x {float(x[0]):6.3f} {float(x[1]):6.3f} | v {float(v[0]):6.3f} {float(v[1]):6.3f} | ee {float(y[0]):6.3f} {float(y[1]):6.3f} | pred {float(y_pred[0]):6.3f} {float(y_pred[1]):6.3f} | err {float(err[0]):6.3f} {float(err[1]):6.3f} | d {float(d[0]):6.3f} {float(d[1]):6.3f} | grad {float(grad[0]):6.3f} {float(grad[1]):6.3f} | vdes {float(vdes[0]):6.3f} {float(vdes[1]):6.3f} | a {float(a[0]):6.3f} {float(a[1]):6.3f}")

        obs,_,_,_,_ = env.step(a)

        if render:
            time.sleep(0.1)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--test', default = False, action ='store_true', 
        help='Test saved model')
    parser.add_argument('--render', default = False, action ='store_true', 
        help='Render')

    args = parser.parse_args()

    if args.test:
        test(args.render)
    else:
        learn(seeds=[1999,1998,1997], learn_steps=300000)
    


