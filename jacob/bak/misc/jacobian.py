import torch 
import torch.nn as nn
import itertools as it
import timeit

from torch.utils.data import Dataset, DataLoader 
from stable_baselines3.common.callbacks import BaseCallback

class Jacob1(nn.Module):
    def __init__(self, n_states, n_hidden_layers, activation, initializer, device):
        super(Jacob1, self).__init__()

        # save these for later use
        self.n_states = n_states
        self.h_dim = n_hidden_layers

        self.j = nn.Sequential()
        
        self.j.append(nn.Linear(3, n_hidden_layers[0]))
        self.j.append(activation())
        
        for n_in, n_out in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            self.j.append(nn.Linear(n_in, n_out))
            self.j.append(activation())
        self.j.append(nn.Linear(n_hidden_layers[-1], n_states))
        
        for l in self.j.children():
            if isinstance(l, nn.Linear):
                initializer(l.weight)
                l.bias.data.fill_(0.0)
        
        self.r = torch.empty((1,), requires_grad=True, device=device)
        return
    
    def parameters(self):
        params = list(self.j.parameters())
        params.append(self.r)
        return params
    
    def forward(self, theta, x_prev):
        _q = torch.hstack((theta, torch.cos(theta), torch.sin(theta)))
        jq = self.j(_q)
        return self.r*jq + x_prev
    
    def state_dict(self):
        sd = dict()
        sd['j'] = self.j.state_dict()
        sd['r'] = self.r
        return sd

    def load_state_dict(self, state_dict):
        self.j.load_state_dict(state_dict['j'])
        self.r = state_dict['r']
        return

class Jacobian(nn.Module):
    def __init__(self, n_joints, n_states, n_hidden_layers, activation = nn.ReLU, initializer = nn.init.xavier_uniform_, device = 'cpu'):
        super(Jacobian, self).__init__()
        
        # save these for later use
        self.device = device
        self.n_joints = n_joints
        self.n_states = n_states
        self.jacobs = [Jacob1(n_states, n_hidden_layers, activation, initializer, self.device) for j in range(n_joints)]

        # concatenate all parameters for the optmizer
        self.params = []
        for j in self.jacobs:
            j.to(self.device)
            self.params = self.params + j.parameters()

        # optimizer
        self._optim_jacobian = torch.optim.SGD(self.params, lr=1e-4)
        return

    def forward(self, q):
        n_samples = len(q)
        y = torch.zeros(n_samples, self.n_states, self.n_joints).to(self.device)
        
        q_acc = torch.cumsum(q, dim=1)

        x_prev = torch.zeros(self.n_states).to(self.device)
        for j in range(self.n_joints):
            _q = q_acc[:,j].reshape(-1,1)
            x_prev = self.jacobs[j](_q, x_prev)
            y[:, :, j] = x_prev
        return y 

    def loss_jacobian(self, x, y):
        n_samples = len(x)
        
        # accumulate cartesian error over all joints and all samples
        acc = torch.zeros(1).to(self.device)
        for s in range(n_samples):
            for v1, v2 in zip(x[s], y[s]):
                acc += torch.norm(v1-v2)
        return acc

    def train_jacobian(self, q, y):
        y_pred = self.forward(q)
        loss = self.loss_jacobian(y_pred, y)
        with torch.no_grad():
            loss.backward()
        self._optim_jacobian.step()
        return loss
    
    def train_from_data(self, data, niter=1000):
        data_loader = DataLoader(dataset=data, batch_size=30, shuffle=True, generator=torch.Generator(device='cpu'))
        
        start = timeit.default_timer()        
        
        mean_loss = 0
        for t in range(niter):
            q, qdot, x = next(iter(data_loader))
            q = q.to(self.device)
            qdot = qdot.to(self.device)
            y = x.to(self.device)

            y_pred = self.forward(q)
            loss = self.loss_jacobian(y_pred, y)
            mean_loss += loss.cpu().detach().numpy()[0]

            self._optim_jacobian.zero_grad()
            loss.backward()
            self._optim_jacobian.step()
            
        end = timeit.default_timer()
        mean_loss /= niter
        print(f"  mean_loss {mean_loss:.6f} | iter {niter} | time {end-start:.2f}")
        
        return mean_loss

    def state_dict(self):
        sd = dict()
        for i, j in enumerate(self.jacobs):
            sd['j%d'%i] = j.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        for i, j in enumerate(self.jacobs):
            j.load_state_dict(state_dict['j%d'%i])
        return

#------------------- Incremental Dataset --------------
class IncrDataset(Dataset):
    def __init__(self, size=0):
        self.size = size
        self.q = []
        self.qdot = []
        self.x = []

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        q = self.q[idx]
        qdot = self.qdot[idx]
        x = self.x[idx]

        return q, qdot, x

    def add(self, q, qdot, x):
        self.q.append(torch.tensor(q, dtype=torch.float32))
        self.qdot.append(torch.tensor(qdot, dtype=torch.float32))
        self.x.append(torch.tensor(x, dtype=torch.float32))
        self.size += 1

# ------------------- ModelLearnCB -------------------

class JacobianLearnCB(BaseCallback):

    def __init__(self, user_quit = [False], verbose: int = 0):
            super().__init__(verbose)
            self.user_quit = user_quit
            self.fknet = None
            self.env = None
            self.fkdata = IncrDataset()
            self.count_rollouts = 0
            print('ModelLearnCB')

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        if self.fknet != None:

            for env in self.env.envs:
                obs = env.get_observation()  # using wrappers
                rew, info = env.get_reward()
                self.fkdata.add(obs['q'], obs['qdot'], obs['x'])
        return not self.user_quit[0]


    def _on_rollout_end(self) -> None:
        self.count_rollouts += 1

        if self.fknet != None:
            mean_loss = self.fknet.train_from_data(self.fkdata, niter=1000)
        return mean_loss

    def _on_training_end(self) -> None:
        pass


