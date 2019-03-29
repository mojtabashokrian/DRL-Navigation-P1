import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import OrderedDict

class DQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size,hidden_layer=2, hidden_layer_size=32, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_layer (int) : Number of hidden layers >= 1 (default = 2)
            hidden_layer_size (int) : Size of hidden layers (default = 32)
            seed (int): Random seed (default = 0)
        """
        super(DQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        myOrderedDict=OrderedDict([])
        myOrderedDict['fc1']=nn.Linear(state_size,hidden_layer_size)
        myOrderedDict['relu1']=nn.ReLU(inplace=True)
        for i in range(hidden_layer-1):
            myOrderedDict[f'fc{i+2}']=nn.Linear(hidden_layer_size,hidden_layer_size)
            myOrderedDict[f'relu{i+2}']=nn.ReLU(inplace=True)
        myOrderedDict[f'fc{hidden_layer+1}']=nn.Linear(hidden_layer_size,action_size)
        self.linear=nn.Sequential(myOrderedDict)
    
    def forward(self, state):
        """maps state -> action values."""
        return self.linear(state)
    
class MHQNetwork(nn.Module):
    """Actor (Policy) Model: Multi Horizon QNetwork """
    def __init__(self, state_size, action_size, mh_size=10, hidden_layer=2, hidden_layer_size=32, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of action
            mh_size (int) : The number of horizons (default = 10)
            hidden_layer (int) : Number of hidden layers >=1 (default = 2)
            hidden_layer_size (int) : Size of hidden layers (default = 32)
            seed (int): Random seed (default = 0)
        """
        super(MHQNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        myOrderedDict=OrderedDict([])
        myOrderedDict['fc1']=nn.Linear(state_size,hidden_layer_size)
        myOrderedDict['relu1']=nn.ReLU(inplace=True)
        for i in range(hidden_layer-1):
            myOrderedDict[f'fc{i+2}']=nn.Linear(hidden_layer_size,hidden_layer_size)
            myOrderedDict[f'relu{i+2}']=nn.ReLU(inplace=True)
        self.linear=nn.Sequential(myOrderedDict)
        self.mh_size=mh_size
        self.mh=nn.ModuleList()
        for i in range(mh_size):
                self.mh.append(nn.Linear(hidden_layer_size,action_size))
    def forward(self, state):
        """a network that maps state -> action values for each horizons all in a torch list."""
        x=self.linear(state)
        if self.mh_size==1: return torch.stack([self.mh[0](x)])
        return torch.stack([self.mh[i](x) for i in range(self.mh_size)]).squeeze(dim=0)
    
class DFQNetwork(nn.Module):
    """ Finding the best discount weighting """
    def __init__(self, action_size, mh_size=10, hidden_layer=0, hidden_layer_size=32, seed=0):
        """Initialize parameters and build model.
        Params
        ======
            action_size (int): Dimension of action
            mh_size (int) : The number of horizons (default = 10)
            hidden_layer (int) : Number of hidden layers, set at 0
            hidden_layer_size (int) : Size of hidden layers (default = 32)
            seed (int): Random seed (default = 0)
        """
        super(DFQNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.action_size=action_size
        self.mh_size=mh_size
        myOrderedDict=OrderedDict([])
        if hidden_layer==0:
            myOrderedDict['fc1']=nn.Linear(self.mh_size,1,bias=False)
        else:
            myOrderedDict['fc1']=nn.Linear(self.mh_size,hidden_layer_size)
            myOrderedDict['relu1']=nn.ReLU(inplace=True)
            for i in range(hidden_layer-1):
                myOrderedDict[f'fc{i+2}']=nn.Linear(hidden_layer_size,hidden_layer_size)
                myOrderedDict[f'relu{i+2}']=nn.ReLU(inplace=True)
            myOrderedDict[f'fc{hidden_layer+1}']=nn.Linear(hidden_layer_size,1,bias=False)
        self.linear=nn.Sequential(myOrderedDict)            
    def forward(self, Qvalues):
        """maps Qvalues (gamma_i,batche_size,actions) of Q_gamma_i(s,a) for all a to Q(s,a)."""
        Qvalues=Qvalues.permute([1,0,2]) 
        return torch.stack([self.linear(Qvalues[:,:,action]) for action in range(self.action_size)]).squeeze(dim=0)