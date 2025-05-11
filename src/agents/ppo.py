import torch
import torch.nn as nn
import torch.optim as optim

from src.models.model import get_ppo_actor, get_ppo_critic

class PPO:
    def __init__(self,
                 actor_cfg,
                 critic_cfg,
                 actor_lr=3e-4,
                 critic_lr=3e-4,
                 gamma=0.99,
                 tau=0.005,
                 epsilon=0.2,
                 update_epochs=10,
                 device="cpu"):
        self.device = device
        
        # Actor, Critic の初期化
        self.actor = get_ppo_actor(actor_cfg=actor_cfg).to(device)
        self.critic = get_ppo_critic(critic_cfg=critic_cfg).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau
        self.epsilon = epsilon
        self.update_epochs = update_epochs

    def select_action(self, state, evaluate=False):
        """アクションの選択"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor(state)
            action = torch.tanh(action)  # 学習時もクリッピング
        return {
            "action": action.cpu().numpy()[0],
            "log_prob": log_prob.cpu().numpy()[0] if log_prob is not None else None
        }

    def update(self, replay_buffer, batch_size=256):
        """ネットワークの更新処理"""
        state, action, reward, next_state, done, old_log_prob, advantage = replay_buffer.sample(batch_size)
        
        # PPO の更新
        a_info = self._update_actor(state, action, old_log_prob, advantage)
        c_info = self._update_critic(state, reward, next_state, done)

        return {**a_info, **c_info}

    def _update_actor(self, state, action, old_log_prob, advantage):
        """Actor の更新処理"""
        for _ in range(self.update_epochs):
            log_prob = self.actor.get_log_prob(state, action)
            ratio = (log_prob - old_log_prob).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * advantage
            loss = -torch.min(surr1, surr2).mean()

            # オプティマイザの更新
            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()
        
        return {'actor_loss': loss.item()}

    def _update_critic(self, state, reward, next_state, done):
        """Critic の更新処理"""
        with torch.no_grad():
            target_value = reward + (1 - done) * self.gamma * self.critic(next_state)
        
        value = self.critic(state)
        loss = nn.MSELoss()(value, target_value)

        # オプティマイザの更新
        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()
        
        return {'critic_loss': loss.item()}

    def save(self, path):
        """モデルの保存"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """モデルのロード"""
        ckpt = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
