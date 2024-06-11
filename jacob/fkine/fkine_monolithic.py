import torch 
import torch.nn as nn

class FKineMono(nn.Module):
    def __init__(self, n_joints, n_dims, n_hidden, size_hidden, lr=1e-4, activation = nn.ReLU, initializer = nn.init.xavier_uniform_, device = 'cpu', model=None):
        super(FKineMono, self).__init__()

        # save these for later use
        self.device = device
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.n_hidden_layers = [size_hidden for h in range(n_hidden)]

        self.fkine = nn.Sequential()
        
        self.fkine.append(nn.Linear(n_joints, self.n_hidden_layers[0]))
        self.fkine.append(activation())
        
        for n_in, n_out in zip(self.n_hidden_layers[:-1], self.n_hidden_layers[1:]):
            self.fkine.append(nn.Linear(n_in, n_out))
            self.fkine.append(activation())
        self.fkine.append(nn.Linear(self.n_hidden_layers[-1], n_joints*n_dims))
        
        for l in self.fkine.children():
            if isinstance(l, nn.Linear):
                initializer(l.weight)
                l.bias.data.fill_(0.0)
        
        self.params = self.fkine.parameters()
        # optimizer
        self._optim = torch.optim.SGD(self.params, lr=lr)

        return
    
    def parameters(self):
        return list(self.fkine.parameters())

    def state_dict(self):
        sd = self.fkine.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        self.fkine.load_state_dict(state_dict)
        return
    
    def forward(self, q):
        if len(q.shape) == 1:
            q = q.reshape(1, len(q))

        n_samples = len(q)
        _x = self.fkine(q)
        x = _x.reshape(n_samples, self.n_dims, self.n_joints)
        return x

    def loss_fkine(self, y_pred, y):
        ret = (y_pred-y).norm(dim=1).mean()
        return ret
        '''
        n_samples = len(y_pred)
        
        # accumulate cartesian error over all joints and all samples
        acc = torch.zeros(1).to(self.device)
        for s in range(n_samples):
            acc += torch.norm(y_pred[s]-y[s], dim=0).sum()
        return acc/self.n_joints
        '''

    def train(self, q, y):
        y_pred = self.forward(q)
        loss = self.loss_fkine(y_pred, y)
        mean_loss = loss.cpu().detach().numpy()
    
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        
        return mean_loss
