import torch 
import torch.nn as nn
from collections import OrderedDict

class FKine1(nn.Module):
    def __init__(self, n_dims, n_hidden_layers, lr, activation, initializer, device):
        super(FKine1, self).__init__()

        # save these for later use
        self.n_dims = n_dims
        self.h_dim = n_hidden_layers
        self.device = device

        self.fkine = nn.Sequential()
        
        self.fkine.append(nn.Linear(3, n_hidden_layers[0]))
        self.fkine.append(activation())

        for n_in, n_out in zip(n_hidden_layers[:-1], n_hidden_layers[1:]):
            self.fkine.append(nn.Linear(n_in, n_out))
            self.fkine.append(activation())
        
        # while DH matrices are 4x4, we only need to compute the 3 first rows 
        self.fkine.append(nn.Linear(n_hidden_layers[-1], 3*4))

        # used for allocating memory in forward()
        self._dh_null = torch.zeros(4, 4)
        self._dh_null[3,3] = 1

        for l in self.fkine.children():
            if isinstance(l, nn.Linear):
                initializer(l.weight)
                l.bias.data.fill_(0.0)
        return

    def forward(self, theta, t_prev):
        n_samples = len(theta)
        _q = torch.hstack((theta, torch.cos(theta), torch.sin(theta)))
        
        # allocate memory for T from prev to this joint
        t = self._dh_null.repeat(n_samples, 1, 1).to(self.device) 
        # allocate memory for T origin to this joint
        t_out = torch.zeros(n_samples, 4, 4).to(self.device)
        _t = self.fkine(_q).reshape(n_samples, 3, 4)
        # mount current T in a 4x4 matrix 
        t[:,:3,:4] = _t 

        # multiply with previous transforms
        t_out = torch.matmul(t_prev, t)
        return t_out, t 

class FKineLinked(nn.Module):
    def __init__(self, n_joints, n_dims, n_hidden, size_hidden, lr=1e-4, activation = nn.Tanh, initializer = nn.init.xavier_uniform_, device = 'cpu', model=None):
        super(FKineLinked, self).__init__()
        # save these for later use
        self.device = device
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.n_hidden_layers = [size_hidden for h in range(n_hidden)]
        
        # instantiate
        for j in range(n_joints):
            self._modules['fkine%d'%j] = FKine1(n_dims, self.n_hidden_layers, lr, activation, initializer, self.device) 
        # DH transformations are always 4x4
        self._t0 = torch.eye(4).to(device)
        
        # optimizer
        self._optim = torch.optim.SGD(self.parameters(), lr=lr)
        return
    
    def forward(self, q):
        if len(q.shape) == 1:
            q = q.reshape(1, len(q))
        
        n_samples = q.shape[0]
        n_joints = q.shape[1]
        y = torch.zeros(n_samples, self.n_dims, n_joints).to(self.device)
        t = torch.zeros(n_samples, n_joints, 4, 4)

        t_prev = self._t0.repeat(n_samples, 1, 1)
        for j in range(n_joints):
            _q = q[:,j].reshape(-1,1)
            t_prev, t[:, j, :, :] = self._modules['fkine%d'%j](_q, t_prev)
            _y = t_prev[:,:3, 3]
            y[:, :, j] = _y[:,:self.n_dims]
        return y, t 

    def loss_fkine(self, y_pred, y):
        ret = (y_pred-y).norm(dim=1).mean()
        return ret

    def loss_rot_matrix(self, t):
        rot = t[:,:,:3,:3]
        l_det = (rot.det()-1).mean()
        
        rott = rot.transpose(2, 3)
        rotxrott = torch.matmul(rot, rott)
        deye = rotxrott-torch.eye(3)
        l_identity = deye.norm(dim=(2,3)).mean()
        ret = l_det + l_identity
        return ret

    def train_from_data(self, q, y):
        y_pred, t = self.forward(q)
        l_kine = self.loss_fkine(y_pred, y)
        l_rot = self.loss_rot_matrix(t)
        loss = l_kine#+l_rot 
        mean_loss = l_kine.cpu().detach().numpy()
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()
        '''
        mean_soss = 0
        #print('link train',q)
        for j in range(self.n_joints):
            #print('joint: ', j)
            #print('tempq', q[:,:j+1])
            y_pred, t = self.forward(q[:,:j+1])
            #print('\n\ntrain\n', y_pred, t)
            l_kine = self.loss_fkine(y_pred, y[:,:,:j+1])
            l_rot = self.loss_rot_matrix(t)
            loss = l_kine+l_rot 
            #input()
            mean_loss += l_kine.cpu().detach().numpy()/self.n_joints
            
            self.fkines[j]._optim.zero_grad()
            loss.backward()
            self.fkines[j]._optim.step()
        '''
        return mean_loss
