import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from src.models.model import get_deterministic_actor, get_critic
from src.utils.helper import soft_update


class TD3:
    def __init__(self,
                 actor_cfg,
                 critic_cfg,
                 gamma=0.99,
                 tau=0.005,
                 policy_noise=0.2,
                 noise_clip=0.5,
                 policy_delay=2,
                 device="cpu"):
        
        self.device = device
        
        # Actor, Critic の初期化
        self.actor = get_deterministic_actor(actor_cfg=actor_cfg).to(device)
        self.actor_target = get_deterministic_actor(actor_cfg=actor_cfg).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # DoubleCritic (q1, q2を持つ) の初期化
        self.critic = get_critic(critic_cfg=critic_cfg)
        self.critic.to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_target.to(device)

        # Optimizer の設定
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_cfg.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_cfg.lr)

        # ハイパーパラメータ
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.update_counter = 0

    def select_action(self, state, evaluate=False, noise=0.1):
        """
        Actor ネットワークからアクションを選択する
        """
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state).cpu().data.numpy().flatten()
        
        if not evaluate and noise != 0:
            action = action + np.random.normal(0, noise, size=action.shape)
        
        action = np.clip(action, -1, 1)

        return {
            "action": action,
            "log_prob": None,  # TD3 では log_prob は使用しない
        }

    def update(self, replay_buffer, batch_size=256):
        """
        TD3 のネットワークの更新
        """
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)

        # Critic の更新
        critic_info = self._update_critic(state, action, reward, next_state, done)
        
        # ポリシーの遅延更新
        actor_info = {}
        if self.update_counter % self.policy_delay == 0:
            actor_info = self._update_actor(state)
            soft_update(self.actor, self.actor_target, self.tau)
            soft_update(self.critic, self.critic_target, self.tau)
        
        self.update_counter += 1

        # Loss を辞書形式で返す
        return {**critic_info, **actor_info}

    def _update_critic(self, state, action, reward, next_state, done):
        """
        Critic ネットワークの更新
        """
        with torch.no_grad():
            # アクションにノイズを付与
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-1, 1)

            # ターゲットQ値の計算
            target_q1, target_q2 = self.critic_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.gamma * target_q

        # 現在のQ値を取得
        current_q1, current_q2 = self.critic(state, action)
        critic_loss1 = nn.MSELoss()(current_q1, target_q)
        critic_loss2 = nn.MSELoss()(current_q2, target_q)
        critic_loss = critic_loss1 + critic_loss2

        # 最適化ステップ
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        return {
            "critic_loss": critic_loss.item(),
            "critic_loss1": critic_loss1.item(),
            "critic_loss2": critic_loss2.item()
        }

    def _update_actor(self, state):
        """
        Actor ネットワークの更新
        """
        # actorが生成したアクションで、criticのq1を評価する
        actor_loss = -self.critic(state, self.actor(state))[0].mean()

        # 最適化ステップ
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return {"actor_loss": actor_loss.item()}

    def save(self, path):
        """
        モデルの保存
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_opt': self.actor_optimizer.state_dict(),
            'critic_opt': self.critic_optimizer.state_dict()
        }, path)

    def load(self, path):
        """
        モデルのロード
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_opt'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_opt'])
