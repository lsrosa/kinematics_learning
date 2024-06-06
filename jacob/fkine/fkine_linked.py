import torch 
import torch.nn as nn

class FKine1(nn.Module):
    def __init__(self, n_dims, n_hidden_layers, lr, activation, prev_params, initializer, device):
        super(FKine1, self).__init__()

        # save these for later use
        self.n_dims = n_dims
        self.h_dim = n_hidden_layers
        self.device = device

        self.fkine = nn.Sequential()
        
        self.fkine.append(nn.Linear(3, n_hidden_layers[0]))
        self.fkine.append(activation())
        # TODO: add initialization
        self.disp = torch.rand(3).to(device).requires_grad_(True)

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

        self._params = prev_params + list(self.fkine.parameters())
        # optimizer
        self._optim = torch.optim.SGD(self._params, lr=lr)
        return
    
    def parameters(self):
        return self._params 
    
    def forward(self, theta, t_prev):
        n_samples = len(theta)
        _q = torch.hstack((theta, torch.cos(theta), torch.sin(theta)))
        
        # allocate memory for T from prev to this joint
        t = self._dh_null.repeat(n_samples, 1, 1).to(self.device) 
        # allocate memory for T origin to this joint
        t_out = torch.zeros(n_samples, 4, 4).to(self.device)
        #print('\nfkine1 q\n', _q)
        _t = self.fkine(_q).reshape(n_samples, 3, 4)
        #print('\nt\n', _t)
        #print('\ndisp\n', self.disp.detach().cpu().numpy())
        # mount current T in a 4x4 matrix 
        t[:,:3,:4] = _t 
        #print('\nt4\n', t)

        # multiply with previous transforms
        t_out = torch.matmul(t, t_prev)
        #print('\ntout\n', t_out)
        return t_out, t 
    
    def state_dict(self):
        sd = dict()
        sd['fkine'] = self.fkine.state_dict()
        sd['disp'] = self.disp
        return sd

    def load_state_dict(self, state_dict):
        self.fkine.load_state_dict(state_dict['fkine'])
        self.disp = state_dict['disp']
        return

class FKineLinked(nn.Module):
    def __init__(self, n_joints, n_dims, n_hidden, size_hidden, lr=1e-4, activation = nn.ReLU, initializer = nn.init.xavier_uniform_, device = 'cpu', model=None):
        super(FKineLinked, self).__init__()
        
        # save these for later use
        self.device = device
        self.n_joints = n_joints
        self.n_dims = n_dims
        self.n_hidden_layers = [size_hidden for h in range(n_hidden)]
        
        # instantiate and set params
        self.fkines = []
        prev_params = []
        for j in range(n_joints):
            self.fkines.append(FKine1(n_dims, self.n_hidden_layers, lr, activation, prev_params, initializer, self.device)) 
            prev_params = self.fkines[-1].parameters()
            self.fkines[-1].to(self.device)
        # DH transformations are always 4x4
        self._t0 = torch.eye(4).to(device)
        return

    def state_dict(self):
        sd = dict()
        for i, j in enumerate(self.fkines):
            sd['fkine_j%d'%i] = j.state_dict()
        return sd

    def load_state_dict(self, state_dict):
        for i, j in enumerate(self.fkines):
            j.load_state_dict(state_dict['fkine_j%d'%i])
        return

    def forward(self, q):
        if len(q.shape) == 1:
            q = q.reshape(1, len(q))
        
        n_samples = q.shape[0]
        n_joints = q.shape[1]
        #print(n_samples, n_joints)
        y = torch.zeros(n_samples, self.n_dims, n_joints).to(self.device)
        t = torch.zeros(n_samples, n_joints, 4, 4)

        t_prev = self._t0.repeat(n_samples, 1, 1)
        for j in range(n_joints):
            _q = q[:,j].reshape(-1,1)
            #print('\nq\n', _q)
            #print('\ntprev\n', t_prev)
            t_prev, t[:, j, :, :] = self.fkines[j](_q, t_prev)
            #print('\ntnew\n', t_prev)
            _y = t_prev[:,:3, 3]
            #print('\nbedfore y\n',y)
            y[:, :, j] = _y[:,:self.n_dims]
            #print('\n after y\n',y)
            #input()
        #print('y', y)
        return y, t 

    def loss_fkine(self, y_pred, y):
        n_samples = len(y_pred)

        # accumulate cartesian error over all joints and all samples
        acc = torch.zeros(1).to(self.device)
        for s in range(n_samples):
            #print(y_pred[s,:]-y[s])
            #print(y_pred[s,:,-1]-y[s,:,-1])
            acc += torch.norm(y_pred[s]-y[s], dim=0).sum()
        return acc/self.n_joints#/n_samples

    def loss_rot_matrix(self, t):
        n_samples = t.shape[0]
        n_joints = t.shape[1]
        #print(t.shape)

        acc = torch.zeros(1).to(self.device)
        rot = t[:,:,:3,:3]
        #print('rot:', rot)
        for s in range(n_samples):
            for j in range(n_joints):
                #print(rot[s,j])
                # the transpose should be the inverse
                acc += torch.norm(torch.matmul(rot[s,j], torch.t(rot[s,j]))-torch.eye(3))
                # determinant should be 1
                acc += torch.norm(torch.det(rot[s,j]))
        return acc/self.n_joints#/n_samples

    def train(self, q, y):
        mean_loss = 0
        #print('link train',q)
        for j in range(self.n_joints):
            #print('tempq', q[:,:j+1])
            y_pred, t = self.forward(q[:,:j+1])
            #print(y_pred, y)
            l_kine = self.loss_fkine(y_pred, y[:,:,:j+1])*2/(self.n_joints+1)
            l_rot = self.loss_rot_matrix(t)
            loss = l_kine+l_rot 
            #input()

            mean_loss += l_kine.cpu().detach().numpy()[0]
            
            self.fkines[j]._optim.zero_grad()
            loss.backward()
            self.fkines[j]._optim.step()
        return mean_loss
