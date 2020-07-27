import numpy as np
import random
from collections import namedtuple, deque
from sum_tree import SumTree

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

from torch.utils.tensorboard import SummaryWriter

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.9 # 0.99           # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

ALPHA = 0.           # level of PER prioritisation
EPSILON = 0. # 0.01         # proportional constant added to each priority # << prev. 0.2 !
BETA_START = 1. # 0.6       # starting value for how much prioritisation to apply
BETA_STEPS = 10000          # number of steps to anneal beta to 1.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, writer):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # TODO: Swap ReplayBuffer for PER buffer
        # Replay memory
#         self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.memory = PrioritisedReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, ALPHA, EPSILON)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        self.beta = BETA_START
        self.writer = writer
        
    
    def step(self, state, action, reward, next_state, done):
        # calculate error, and store experience in replay buffer accordingly
        
#         next_actions = self.qnetwork_local(next_states).max(1).indices.unsqueeze(1)
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1).values.unsqueeze(1) # << [64,1] of max Q values
#         Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
#         Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
#         Q_expected = self.qnetwork_local(states).gather(1, actions) # gather uses the actions as indices to select the Qs
        # get next action from qnetwork_local, using next_state
        # get next reward using next action, from qnetwork_target
        # calc. target: reward + (gamma * next reward * done mask)
        # get expected from qnetwork_local, for current state/action
        s = torch.tensor([state]).float().to(device)
        ns = torch.tensor([next_state]).float().to(device)
        
        next_action = self.qnetwork_local(ns).max(1).indices.unsqueeze(1)
        next_reward = self.qnetwork_target(ns).cpu().detach().numpy()[0, next_action]
        
        target = reward + (GAMMA * next_reward * (1 - done))
        expected = self.qnetwork_local(s).cpu().detach().numpy()[0, action]
        error = abs(target - expected)
        
        self.memory.add(error, state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step += 1 
        
        self.writer.add_scalar('Timestep Error', error, self.t_step)
        
        if (self.t_step % UPDATE_EVERY) == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                self.beta += ((1 - self.beta) / BETA_STEPS) # anneal the beta, from a starting value towards 1.0
                self.beta = np.min([1., self.beta])
                
                experiences = self.memory.sample(self.beta)
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
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
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
#         states, actions, rewards, next_states, dones = experiences
        states, actions, rewards, next_states, dones, weights, idxs = experiences
        
        next_actions = self.qnetwork_local(next_states).max(1).indices.unsqueeze(1)
        
        
#         Q_targets_next = self.qnetwork_target(next_states).detach().max(1).values.unsqueeze(1) # << [64,1] of max Q values
        Q_targets_next = self.qnetwork_target(next_states).detach().gather(1, next_actions)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        Q_expected = self.qnetwork_local(states).gather(1, actions) # gather uses the actions as indices to select the Qs
        
        # refresh errors in replay buffer
        errors = torch.abs(Q_expected - Q_targets).cpu().detach().numpy()
        for (idx, error) in zip(idxs, errors):
            self.memory.update(idx, error)
        
        loss = (weights.detach() * F.mse_loss(Q_expected, Q_targets)).mean() # weighted loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

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

class PrioritisedReplayBuffer:
    def __init__(self, action_size, buffer_size, batch_size, alpha, epsilon):
        self.action_size = action_size
        self.tree = SumTree(buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.alpha = alpha
        self.epsilon = epsilon
    
    def add(self, error, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        p = self._get_priority(error)
        self.tree.add(p, e)
    
    def sample(self, beta):
        segment = self.tree.total() / self.batch_size # split into segments so we don't end up with duplicates innit
        
        experiences = []
        priorities = []
        idxs = []
        
        for i in range(self.batch_size):
            start = segment * i
            end = segment * (i + 1)
            s = random.uniform(start, end)
            idx, p, e = self.tree.get(s)
            if e:
                priorities.append(p)
                experiences.append(e)
                idxs.append(idx)
        
        probs = priorities / self.tree.total() # big P
        weights = np.power(self.tree.n_entries * probs, -beta)
        weights /= weights.max() # scale so max weight is 1
      
        states = torch.from_numpy(np.vstack([e.state for e in experiences])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences]).astype(np.uint8)).float().to(device)
        weights = torch.from_numpy(weights).float().to(device)
        
        return (states, actions, rewards, next_states, dones, weights, idxs)
    
    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)
    
    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha
    
    def __len__(self):
        """Return the current size of internal memory."""
        return self.tree.n_entries