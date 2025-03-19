import torch
import torch.nn as nn

def gaussian_init_(n_units, std=1):    
    sampler = torch.distributions.Normal(torch.Tensor([0]), torch.Tensor([std/n_units]))
    Omega = sampler.sample((n_units, n_units))[..., 0]  
    return Omega

"""
Residual Network
"""
class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, encode_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, encode_dim)

        self.residual_connection = nn.Linear(input_dim, encode_dim) if input_dim != encode_dim else nn.Identity()

    def forward(self, x):
        residual = self.residual_connection(x)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out += residual
        return self.relu(out)

class ResidualNetwork(nn.Module): 
    def __init__(self, Nkoopman, u_dim, input_dim, encode_dim, hidden_dim=256, num_blocks=3):
        super(ResidualNetwork, self).__init__()
        self.initial_fc = nn.Linear(input_dim, hidden_dim)
        self.blocks = nn.ModuleList([ResidualBlock(hidden_dim, hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.final_fc = nn.Linear(hidden_dim, encode_dim)

        self.Nkoopman = Nkoopman
        self.u_dim = u_dim 
        self.s_dim = input_dim

        self.lA = nn.Linear(Nkoopman,Nkoopman,bias=False)
        self.lA.weight.data = gaussian_init_(Nkoopman, std=1)
        U, _, V = torch.svd(self.lA.weight.data)
        self.lA.weight.data = torch.mm(U, V.t()) * 0.9
        self.lB = nn.Linear(u_dim,Nkoopman,bias=False)

    def encode_only(self, x):
        x = self.initial_fc(x)
        for block in self.blocks:
            x = block(x)
        x = self.final_fc(x)
        return x

    def encode(self, x):
        return torch.cat([x,self.encode_only(x)],axis=-1)
    
    def forward(self, z, u):
        return self.lA(z)+self.lB(u)