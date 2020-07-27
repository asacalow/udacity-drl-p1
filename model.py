import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import OrderedDict 

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
#         layer_sizes = np.array([512,256,128,64])
        
#         for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
#             print(in_size, out_size)
        
#         self.model = nn.Sequential(OrderedDict([
#            ('fc1', nn.Linear(state_size, 128)),
#            ('relu1', nn.ReLU()),
#            ('fc2', nn.Linear(128, 128)),
#            ('relu2', nn.ReLU()),
#            ('fc3', nn.Linear(128, action_size))
#         ]))
        self.model = nn.Sequential(OrderedDict([
           ('fc1', nn.Linear(state_size, 256)),
           ('relu1', nn.ReLU()),
           ('fc2', nn.Linear(256, 128)),
           ('relu2', nn.ReLU()),
           ('fc3', nn.Linear(128, action_size))
        ]))
                                                          

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.model(state)
    
class DuelingQNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DuelingQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
#         layer_sizes = np.array([512,256,128,64])
        
#         for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
#             print(in_size, out_size)
        
#         self.model = nn.Sequential(OrderedDict([
#            ('fc1', nn.Linear(state_size, 128)),
#            ('relu1', nn.ReLU()),
#            ('fc2', nn.Linear(128, 128)),
#            ('relu2', nn.ReLU()),
#            ('fc3', nn.Linear(128, action_size))
#         ]))
        inner_size = 128

        self.features = nn.Sequential(OrderedDict([
           ('f-fc1', nn.Linear(state_size, inner_size)),
           ('f-relu1', nn.ReLU()),
           ('f-fc2', nn.Linear(inner_size, inner_size)),
           ('f-relu2', nn.ReLU())
        ]))
        self.values = nn.Sequential(OrderedDict([
           ('v-fc1', nn.Linear(inner_size, inner_size)),
           ('v-relu1', nn.ReLU()),
           ('v-fc2', nn.Linear(inner_size, 1))
        ]))
        self.advantages = nn.Sequential(OrderedDict([
           ('a-fc1', nn.Linear(inner_size, inner_size)),
           ('a-relu1', nn.ReLU()),
           ('a-fc2', nn.Linear(inner_size, action_size))
        ]))
                                                          

    def forward(self, state):
        """Build a network that maps state -> action values."""
        f = self.features(state)
        v = self.values(f)
        a = self.advantages(f)
        
        return ( v + a - a.mean() )

