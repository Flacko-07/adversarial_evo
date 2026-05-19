import torch 
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, noise_dim, cond_dim, output_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        in_dim = noise_dim + cond_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, noise, condition):
        x = torch.cat([noise, condition], dim=1)
        return self.net(x)

class Critic(nn.Module):
    def __init__(self, input_dim, cond_dim, hidden_dims=[128, 64]):
        super().__init__()
        layers = []
        in_dim = input_dim + cond_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, 1))  # No activation for WGAN critic
        self.net = nn.Sequential(*layers)

    def forward(self, x, condition):
        x = torch.cat([x, condition], dim=1)
        return self.net(x)
