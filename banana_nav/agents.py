from typing import Dict, Any, Tuple, NamedTuple, Iterable, List
import numpy as np
import random
from collections import deque

import torch
import torch.nn.functional as F
from torch import optim

from banana_nav.model import QNetwork

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DQNAgent:
    """Interacts with and learns from the environment."""

    def __init__(
            self,
            state_size: int,
            action_size: int,
            buffer_size: int = 100_000,
            batch_size: int = 64,
            gamma_discount_factor: float = 0.99,
            tau_soft_update: float = 1e-3,
            learning_rate: float = 5e-4,
            update_network_every: int = 4,
            seed: int = 0):
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
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.update_network_every = update_network_every
        self.gamma_discount_factor = gamma_discount_factor
        self.tau_soft_update = tau_soft_update

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)

        # Replay memory
        self.memory = ReplayBuffer(action_size, buffer_size, batch_size, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            'seed': self.seed,
            'action_size': self.action_size,
            'state_size': self.state_size,
            'q_network_local': self.qnetwork_local.metadata,
            'q_network_target': self.qnetwork_local.metadata,
        }

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % self.update_network_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.gamma_discount_factor)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(
            self,
            experiences: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
            gamma: float):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences: tuple of (s, a, r, s', done) tuples
            gamma: discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        _soft_update(self.qnetwork_local, self.qnetwork_target, self.tau_soft_update)


def _soft_update(local_model, target_model, tau):
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


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(
            self,
            action_size: int,
            buffer_size: int,
            batch_size: int,
            seed: int):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size: dimension of each action
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
            seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = Experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(_drop_none(states))).float().to(DEVICE)
        actions = torch.from_numpy(np.vstack(_drop_none(actions))).long().to(DEVICE)
        rewards = torch.from_numpy(np.vstack(_drop_none(rewards))).float().to(DEVICE)
        next_states = torch.from_numpy(np.vstack(_drop_none(next_states))).float().to(DEVICE)
        dones = torch.from_numpy(np.vstack(_drop_none(dones)).astype(np.uint8)).float().to(DEVICE)
  
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)


def _drop_none(xs: Iterable[Any]) -> List[Any]:
    return [x for x in xs if x is not None]


class Experience(NamedTuple):
    state: np.array
    action: int
    reward: float
    next_state: np.array
    done: bool

