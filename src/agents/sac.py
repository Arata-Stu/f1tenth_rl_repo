import torch
import torch.nn as nn
import torch.optim as optim

from src.models.actor import GaussianPolicy, Gaussian1dConvPolicy
from src.models.critic import DoubleCritic, Double1dConvCritic


class SAC:
    def __init__(self,
                 state_dim,
                 action_dim,
                 hidden_dim=256,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 alpha_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 target_entropy=None,
                 device="cpu"):
        self.device = device
        # Actor, Critic
        self.actor = Gaussian1dConvPolicy(state_dim, action_dim, hidden_dim).to(device)
        self.critic = Double1dConvCritic(state_dim, action_dim, hidden_dim, tau).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # α (温度パラメータ)
        if target_entropy is None:
            target_entropy = -action_dim
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
        self.target_entropy = target_entropy

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if evaluate:
                mean, _ = self.actor(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        
        c_info = self._update_critic(state, action, reward, next_state, done)
        a_info = self._update_actor(state)
        alpha_info = self._update_alpha(state)
        self.critic.soft_update()

        return {**c_info, **a_info, **alpha_info}

    def _update_critic(self, state, action, reward, next_state, done):
        with torch.no_grad():
            na, nlp = self.actor.sample(next_state)
            alpha = self.log_alpha.exp()
            tq1, tq2 = self.critic.target(next_state, na)
            q_target = torch.min(tq1, tq2) - alpha * nlp
            target_q = reward + (1 - done) * self.gamma * q_target

        q1, q2 = self.critic.forward(state, action)
        loss = nn.MSELoss()(q1, target_q) + nn.MSELoss()(q2, target_q)
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        return {'critic_loss': loss.item()}

    def _update_actor(self, state):
        na, lp = self.actor.sample(state)
        q1, q2 = self.critic.forward(state, na)
        q_min = torch.min(q1, q2)
        alpha = self.log_alpha.exp()
        loss = (alpha * lp - q_min).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        self.actor_optimizer.step()
        return {'actor_loss': loss.item()}

    def _update_alpha(self, state):
        _, lp = self.actor.sample(state)
        loss = -(self.log_alpha * (lp + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        loss.backward()
        self.alpha_optimizer.step()
        return {'alpha_loss': loss.item(), 'alpha_value': self.log_alpha.exp().item()}

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic1': self.critic.critic1.state_dict(),
            'critic2': self.critic.critic2.state_dict(),
            'log_alpha': self.log_alpha,
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict(),
            'alpha_opt': self.alpha_optimizer.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.critic1.load_state_dict(ckpt['critic1'])
        self.critic.critic2.load_state_dict(ckpt['critic2'])
        # ターゲットをコピー
        self.critic.target1.load_state_dict(self.critic.critic1.state_dict())
        self.critic.target2.load_state_dict(self.critic.critic2.state_dict())
        self.log_alpha = ckpt['log_alpha']
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.alpha_optimizer.load_state_dict(ckpt['alpha_opt'])
