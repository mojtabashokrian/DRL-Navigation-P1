import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hyperparameters import *
from models import *
from ReplayBuffer import *

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, network = "mhq_dfq",
                 mh_size=10, hidden_layer=2,hidden_layer_d=0, hidden_layer_size=32, time_aware=False,  seed=SEED):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            gamma: discount factor, set at 0.99
            network: "mhq_dfq" for multihorizon, else vanilla DQNetwork 
            mh_size: for MHQNetwork, set at 10
            hidden_layer: for MHQNetwork, set at 2
            hidden_layer_d: for DFQNetwork, set at 0
            hidden_layer_size: for MHQNetwork, set at 32
            seed (int): random seed default 0
        """
        self.time_aware=time_aware
        if self.time_aware==True:
            self.state_size = state_size+1
        else:
            self.state_size = state_size
        self.network=network
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.hidden_layer=hidden_layer
        self.hidden_layer_size=hidden_layer_size
        self.seed=seed
        self.gamma=GAMMA
        self.lr=LR
        self.tau=TAU
        if self.network=="mhq_dfq":
            self.hidden_layer_d=hidden_layer_d
            self.mh_size=mh_size
            # MHQ,DFQ-Networks
            self.mhqnetwork_local = MHQNetwork(self.state_size, action_size, mh_size, hidden_layer, hidden_layer_size, seed).to(device)
            self.mhqnetwork_target = MHQNetwork(self.state_size, action_size, mh_size, hidden_layer, hidden_layer_size, seed).to(device)
            self.dfqnetwork_local = DFQNetwork(action_size, mh_size, hidden_layer_d, hidden_layer_size,seed).to(device)
            self.dfqnetwork_target = DFQNetwork(action_size, mh_size, hidden_layer_d, hidden_layer_size, seed).to(device)
            self.optimizer_mhq = optim.Adam(self.mhqnetwork_local.parameters(), lr=self.lr)
            self.optimizer_dfq = optim.Adam(self.dfqnetwork_local.parameters(), lr=self.lr)
            #discounts for multi horizons
            self.discounts=torch.tensor([self.gamma**(self.mh_size-i) for i in range(self.mh_size)]).to(device)
            
        else:
            # vanilla DQNetwork
            self.qnetwork_local=DQNetwork(self.state_size, action_size, hidden_layer, hidden_layer_size, seed).to(device)
            self.qnetwork_target=DQNetwork(self.state_size, action_size, hidden_layer, hidden_layer_size, seed).to(device)
            self.optimizer_q = optim.Adam(self.qnetwork_local.parameters(), lr=self.lr)
        #choosing Huber Loss
        self.criterion =  F.smooth_l1_loss 
        
        # Replay memory
        self.batch_size=BATCH_SIZE
        self.buffer_size=BUFFER_SIZE
        self.memory = ReplayBuffer(action_size, self.buffer_size, self.batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.update_every=UPDATE_EVERY
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done, time_step=None):
        # Save experience in replay memory
        if self.time_aware==True: 
            self.memory.add(np.append(state,time_step/300), action, reward, np.append(next_state,(time_step+1)/300) , done)
        else:
            self.memory.add(state, action, reward, next_state, done)
        self.t_step = (self.t_step + 1) % self.update_every                
        if self.t_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma)

    def act(self, state, eps=0.,mh_take_action=False,which_mh=None, time_step=None):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
            mh_take_action (bool): a branch of MultiHorizon takes action, set at False
            which_mh (int): between 0 to mh_size-1, which MH branch should take action; mh_take_action should be
            True for this to work
            time_step (int): the time_step of the environment (used for implementing time awareness)
        """
        if self.time_aware==True:
            state = torch.from_numpy(np.append(state,time_step/300)).float().unsqueeze(0).to(device)
        else:
            state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        if self.network=="mhq_dfq":
            self.mhqnetwork_local.eval()
            self.dfqnetwork_local.eval()
            with torch.no_grad():
                mh_action_values = self.mhqnetwork_local(state)
                action_values = self.dfqnetwork_local(mh_action_values)
            self.mhqnetwork_local.train()
            self.dfqnetwork_local.train()
            # Epsilon-greedy action selection
            if random.random() > eps:
                if mh_take_action==True:  
                    return np.argmax(mh_action_values[which_mh].cpu().data.numpy())
                else:
                    return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train()
            # Epsilon-greedy action selection
            if random.random() > eps:
                return np.argmax(action_values.cpu().data.numpy())
            else:
                return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        if self.network=="mhq_dfq": 
            self.mhq_learn(experiences, gamma)
        else:
            self.q_learn(experiences, gamma)      
        
    def mhq_learn(self, experiences, gamma):  
        states, actions, rewards, next_states, dones = experiences
        self.optimizer_mhq.zero_grad()
        self.optimizer_dfq.zero_grad()
        self.mhqnetwork_local.train()        
        self.dfqnetwork_local.train()
        self.mhqnetwork_target.eval()
        self.dfqnetwork_target.eval()
        #computing the target values
        with torch.no_grad():
            indices_non_terminal=torch.tensor(1-torch.squeeze(dones,dim=1),dtype=torch.uint8)
            non_terminal=next_states[indices_non_terminal]
            #mhq target values
            Qvalues_non_terminal_next_state=self.mhqnetwork_target(non_terminal)
            MAXMHQ_next_action_values=torch.zeros((self.mh_size,)+actions.shape,dtype=torch.float).to(device)
            for i in range(self.mh_size):
                MAXMHQ_next_action_values[i][indices_non_terminal]=torch.max(Qvalues_non_terminal_next_state[i],
                                                                             dim=1)[0].unsqueeze(dim=1).detach()
            #dfq target values
            dfq_next_action_values=self.dfqnetwork_target(Qvalues_non_terminal_next_state)
            MAXDFQ_next_action_values=torch.zeros_like(actions,dtype=torch.float).to(device)
            MAXDFQ_next_action_values[indices_non_terminal]=torch.max(dfq_next_action_values,dim=0)[0].detach()
        #computing the prediction values mhq
        Qvalues=self.mhqnetwork_local(states)       
        loss_mhq=np.array([self.criterion(Qvalues[i].gather(1,actions),
                               rewards+self.discounts[i]*MAXMHQ_next_action_values[i]) for i in range(self.mh_size)])
        loss_mhq=loss_mhq.sum()
        loss_mhq.backward()                                        
        self.optimizer_mhq.step()
        #computing the prediction values dfq
        Qvalues=Qvalues.detach().requires_grad_(False)
        dfq_values=self.dfqnetwork_local(Qvalues)
        dfq_values=dfq_values.permute([2,1,0]).squeeze().gather(1,actions)
        loss_dfq=self.criterion(dfq_values,rewards+MAXDFQ_next_action_values)
        loss_dfq.backward()
        self.optimizer_dfq.step() 
        #---------------------update target nets----------------------- #
        self.soft_update(self.mhqnetwork_local, self.mhqnetwork_target, self.tau)                     
        self.soft_update(self.dfqnetwork_local, self.dfqnetwork_target, self.tau)
        
    def q_learn(self, experiences, gamma): 
        states, actions, rewards, next_states, dones = experiences
        self.optimizer_q.zero_grad()
        self.qnetwork_local.train()
        #computing target values
        next_action_values=torch.zeros_like(actions,dtype=torch.float)
        indices_non_terminal=torch.tensor(1-torch.squeeze(dones,dim=1),dtype=torch.uint8)
        non_terminal=next_states[indices_non_terminal]
        self.qnetwork_target.eval()
        next_action_values[indices_non_terminal]=torch.max(self.qnetwork_target(non_terminal),dim=1)[0].unsqueeze(dim=1).detach()
        #computing the prediction values
        action_values=self.qnetwork_local(states).gather(1, actions)
        loss=self.criterion(action_values,rewards+gamma*next_action_values)
        loss.backward()
        self.optimizer_q.step()
        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.tau)     
        
    def soft_update(self, local_model, target_model, tau): 
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)