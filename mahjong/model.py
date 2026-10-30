"""Small portable CNN. The deployment process imports no RL framework."""

import torch
from torch import nn


class Model(nn.Module):
    def __init__(self, hidden=64):
        super(Model, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(6, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
        )
        self.hidden = nn.Sequential(nn.Linear(16 * 36, hidden), nn.ReLU())
        self.actor, self.critic = nn.Linear(hidden, 235), nn.Linear(hidden, 1)

    def forward(self, obs):
        x = self.hidden(self.conv(obs).reshape(obs.shape[0], -1))
        return self.actor(x), self.critic(x).squeeze(-1)
