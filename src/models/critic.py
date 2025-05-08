import copy
import torch
import torch.nn as nn

from .backbone import TinyLidarBackbone

class DoubleCritic(nn.Module):
    """
    二重Criticネットワークとターゲットの管理
    """
    def __init__(self, state_dim, action_dim, hidden_dim, tau, device):
        super().__init__()
        self.device = device
        # Q1, Q2 ネットワーク
        def make_net():
            return nn.Sequential(
                nn.Linear(state_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.critic1 = make_net().to(device)
        self.critic2 = make_net().to(device)
        # ターゲットネットワーク
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)
        self.tau = tau

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2

    def target(self, state, action):
        x = torch.cat([state, action], dim=1)
        tq1 = self.target1(x)
        tq2 = self.target2(x)
        return tq1, tq2

    def soft_update(self):
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)


class Double1dConvCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, tau, device):
        super().__init__()
        self.device = device
        self.lidar_backbone = TinyLidarBackbone(input_dim=state_dim)
        def make_net():
            return nn.Sequential(
                nn.Linear(self.lidar_backbone.out_dim + action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        self.critic1 = make_net().to(device)
        self.critic2 = make_net().to(device)
        self.target1 = copy.deepcopy(self.critic1)
        self.target2 = copy.deepcopy(self.critic2)
        self.tau = tau

    def forward(self, state, action):
        features = self.lidar_backbone(state)
        x = torch.cat([features, action], dim=1)
        q1 = self.critic1(x)
        q2 = self.critic2(x)
        return q1, q2

    def target(self, state, action):
        features = self.lidar_backbone(state)
        x = torch.cat([features, action], dim=1)
        tq1 = self.target1(x)
        tq2 = self.target2(x)
        return tq1, tq2

    def soft_update(self):
        for p, tp in zip(self.critic1.parameters(), self.target1.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic2.parameters(), self.target2.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

