import torch
import torch.nn as nn
from torch.distributions import Categorical


class PPOModel(nn.Module):
    def __init__(self, obs_dim=2, hidden=128, n_actions=2):
        super().__init__()

        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, n_actions),  # logits
        )

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def policy(self, x):
        logits = self.actor(x)
        return Categorical(logits=logits)

    def value(self, x):
        return self.critic(x)
