import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.body = nn.Sequential(*layers)
        self.mu_head = nn.Linear(last_dim, act_dim)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def forward(self, x):
        x = self.body(x)
        mu = self.mu_head(x)
        std = torch.exp(self.log_std)
        return mu, std


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        last_dim = obs_dim
        for h in hidden_sizes:
            layers += [nn.Linear(last_dim, h), nn.ReLU()]
            last_dim = h
        self.body = nn.Sequential(*layers)
        self.v_head = nn.Linear(last_dim, 1)

    def forward(self, x):
        x = self.body(x)
        v = self.v_head(x)
        return v
