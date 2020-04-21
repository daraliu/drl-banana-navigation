import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(
            self,
            state_size: int,
            action_size: int,
            seed: int,
            fc1_units: int = 64,
            fc2_units: int = 64):
        """Initialize parameters and build model.
        Params
        ======
            state_size : Dimension of each state
            action_size : Dimension of each action
            seed : Random seed
            fc1_units : Number of nodes in first hidden layer
            fc2_units : Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1)
