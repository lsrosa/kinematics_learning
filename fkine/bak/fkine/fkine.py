import torch 
import torch.nn as nn
import itertools as it
import timeit

from torch.utils.data import Dataset, DataLoader 
from stable_baselines3.common.callbacks import BaseCallback

class FKine1(nn.Module):
    def __init__(self, n_states, n_hidden_layers, activation, initializer, device):
        super(FKine1, self).__init__()

        # save these for later use
        self.n_states = n_states
        self.h_dim = n_hidden_layers

        self.fkine = nn.Sequential()
        
        self.fkine.append(nn.Linear(3, n_hidden_layers[0]))
        self.fkine.append(activation())
        
        for n_in, n_out in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            self.fkine.append(nn.Linear(n_in, n_out))
            self.fkine.append(activation())
        self.fkine.append(nn.Linear(n_hidden_layers[-1], n_states))
        
        for l in self.fkine.children():
            if isinstance(l, nn.Linear):
                initializer(l.weight)
                l.bias.data.fill_(0.0)
        return
    
    def parameters(self):
        return list(self.fkine.parameters())
    
    def forward(self, theta, x_prev):
        _q = torch.hstack((theta, torch.cos(theta), torch.sin(theta)))
        x = self.fkine(_q)
        return x + x_prev
    
    def state_dict(self):
        sd = dict()
        sd['fkine'] = self.fkine.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        self.fkine.load_state_dict(state_dict['fkine'])
        return

class FKine(nn.Module):
    def __init__(self, n_joints, n_states, n_hidden_layers, activation = nn.ReLU, initializer = nn.init.xavier_uniform_, device = 'cpu'):
        super(FKine, self).__init__()
        
        # save these for later use
        self.device = device
        self.n_joints = n_joints
        self.n_states = n_states
        self.fkines = [FKine1(n_states, n_hidden_layers, activation, initializer, self.device) for j in range(n_joints)]

        # concatenate all parameters for the optmizer
        self.params = []
        for j in self.fkines:
            j.to(self.device)
            self.params = self.params + j.parameters()

        # optimizer
        self._optim = torch.optim.SGD(self.params, lr=1e-4)
        return

    def forward(self, q):
        if len(q.shape) == 1:
            q = q.reshape(1, len(q))
        n_samples = len(q)
        y = torch.zeros(n_samples, self.n_states, self.n_joints).to(self.device)
        
        q_acc = torch.cumsum(q, dim=1)

        x_prev = torch.zeros(self.n_states).to(self.device)
        for j in range(self.n_joints):
            _q = q_acc[:,j].reshape(-1,1)
            x_prev = self.fkines[j](_q, x_prev)
            y[:, :, j] = x_prev
        return y 

    def loss_fkine(self, y_pred, y):
        n_samples = len(y_pred)
        
        # accumulate cartesian error over all joints and all samples
        acc = torch.zeros(1).to(self.device)
        for s in range(n_samples):
            for x1, x2 in zip(y_pred[s], y[s]):
                acc += torch.norm(x1-x2)
        return acc
    
    def train(self, data, niter=1000):
        data_loader = DataLoader(dataset=data, batch_size=30, shuffle=True, generator=torch.Generator(device='cpu'))
        
        start = timeit.default_timer()        
        
        mean_loss = 0
        for t in range(niter):
            q, qdot, x, xdot = next(iter(data_loader))
            q = q.to(self.device)
            y = x.to(self.device)

            y_pred = self.forward(q)
            loss = self.loss_fkine(y_pred, y)
            mean_loss += loss.cpu().detach().numpy()[0]

            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
            
        end = timeit.default_timer()
        mean_loss /= niter
        print(f"  fkine mean_loss {mean_loss:.6f} | iter {niter} | time {end-start:.2f}")
        
        return mean_loss

    def state_dict(self):
        sd = dict()
        for i, j in enumerate(self.fkines):
            sd['fkine_j%d'%i] = j.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        for i, j in enumerate(self.fkines):
            j.load_state_dict(state_dict['fkine_j%d'%i])
        return

#------------------- Incremental Dataset --------------
class IncrDataset(Dataset):
    def __init__(self, size=0):
        self.size = size
        self.q = []
        self.qdot = []
        self.x = []
        self.xdot = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        q = self.q[idx]
        qdot = self.qdot[idx]
        x = self.x[idx]
        xdot = self.xdot[idx]
        return q, qdot, x, xdot

    def add(self, q, qdot, x, xdot):
        self.q.append(torch.tensor(q, dtype=torch.float32))
        self.qdot.append(torch.tensor(qdot, dtype=torch.float32))
        self.x.append(torch.tensor(x, dtype=torch.float32))
        self.xdot.append(torch.tensor(xdot, dtype=torch.float32))
        self.size += 1

# ------------------- ModelLearnCB -------------------

class LearnCB(BaseCallback):
    def __init__(self, user_quit = [False], verbose: int = 0):
            super().__init__(verbose)
            self.user_quit = user_quit
            self.fkine_net = None
            self.jacob_net = None
            self.env = None
            self.data = None 
            self.count_rollouts = 0
            print('ModelLearnCB')

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.fkine_net != None:
            for env in self.env.envs:
                obs = env.get_observation()  # using wrappers
                rew, info = env.get_reward()
                self.data.add(obs['q'], obs['qdot'], obs['x'], obs['xdot'])
        return not self.user_quit[0]


    def _on_rollout_end(self) -> None:
        fkine_loss = None
        jacob_loss = None
        self.count_rollouts += 1

        if self.fkine_net != None:
            fkine_loss = self.fkine_net.train(self.data, niter=1000)
        if self.jacob_net != None:
            jacob_loss = self.jacob_net.train(self.data, niter=1000)

        return fkine_loss, jacob_loss 

    def _on_training_end(self) -> None:
        pass


